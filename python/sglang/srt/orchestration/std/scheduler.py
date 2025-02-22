import faulthandler
import logging
import os
import signal
from types import SimpleNamespace
from typing import List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    FlushCacheReq,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    broadcast_pyobj,
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class SchedulerCommunicator:
    def __init__(
        self,
        core: Scheduler,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.core = core
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.Context(2)
        if self.core.attn_tp_rank == 0:
            self._recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self._send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

            if server_args.skip_tokenizer_init:
                # Directly send to the StdOrchestrator
                self._send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self._send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )
        else:
            self._recv_from_tokenizer = None
            self._send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self._send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.core.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.core.handle_embedding_request),
                (FlushCacheReq, self.core.flush_cache_wrapped),
                (AbortReq, self.core.abort_request),
                (UpdateWeightFromDiskReqInput, self.core.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.core.init_weights_update_group),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.core.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.core.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.core.get_weights_by_name),
                (ProfileReq, self.core.profile),
                (OpenSessionReqInput, self.core.open_session),
                (CloseSessionReqInput, self.core.close_session),
                (
                    ReleaseMemoryOccupationReqInput,
                    lambda _: self.core.release_memory_occupation(),
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    lambda _: self.core.resume_memory_occupation(),
                ),
            ]
        )

        core.on_generation_output = self._handle_generation_output

    def recv_and_process_input_requests(self):
        self._process_input_requests(self._recv_requests())

    def _recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.core.attn_tp_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self._recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.server_args.enable_dp_attention:
            if self.core.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.core.attn_tp_size != 1:
                attn_tp_rank_0 = self.core.dp_rank * self.core.attn_tp_size
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.core.attn_tp_rank,
                    self.core.attn_tp_cpu_group,
                    src=attn_tp_rank_0,
                )
            if self.core.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs, self.core.tp_rank, self.core.tp_cpu_group
                )
            recv_reqs = work_reqs + control_reqs
        elif self.core.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs, self.core.tp_rank, self.core.tp_cpu_group
            )
        return recv_reqs

    def _process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            output = self._request_dispatcher(recv_req)
            if output is not None:
                self._send_to_tokenizer.send_pyobj(output)

    def _handle_generation_output(self, obj):
        self._send_to_detokenizer.send_pyobj(obj)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    setproctitle.setproctitle("sglang::scheduler")
    faulthandler.enable()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configue the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" DP{dp_rank} TP{tp_rank}")
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    parent_process = psutil.Process().parent()

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args, port_args.nccl_port, gpu_id, tp_rank, dp_rank
        )
        communicator = SchedulerCommunicator(
            core=scheduler,
            server_args=server_args,
            port_args=port_args,
        )

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        while True:
            communicator.recv_and_process_input_requests()
            scheduler.process_batch()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
