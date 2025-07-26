from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    moe_ep_deepgemm_preprocess,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_cutlass_moe_ep_preproess,
    run_moe_ep_preproess,
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_triton_kernel,
    tma_align_input_scale,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.srt.utils import dispose_tensor

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class GroupedGemmRunner(torch.nn.Module):

    flashinfer_gemm_warpper = None
    _runner = None

    def __init__(
        self,
        device,
        use_flashinfer: bool = False,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        self.use_per_token_if_dynamic = use_per_token_if_dynamic
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def get_runner(cls, device, use_flashinfer: bool = False, use_per_token_if_dynamic: bool = True):
        if cls._runner is None:
            cls._runner = GroupedGemmRunner(device, use_flashinfer, use_per_token_if_dynamic)
        return cls._runner

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
        c_dtype=None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
                c_dtype=c_dtype,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        return c


grouped_gemm_runner = None


def fused_ep_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_output: TopKOutput,
    *,
    start_expert_id: int,
    end_expert_id: int,
    activation: str = "silu",
    activation_scheme: Optional[str] = None,
    use_fp8_w8a8: bool = False,
    use_block_quant: bool = False,
    use_per_token_if_dynamic: bool = True,
    w13_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w13_input_scale: Optional[torch.Tensor] = None,
    w2_input_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
):

    topk_weights, topk_ids, _ = topk_output

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    grouped_gemm_runner = GroupedGemmRunner.get_runner(
        hidden_states.device,
        use_flashinfer=False,  # TODO: use flashinfer
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )

    top_k = topk_ids.shape[1]
    num_experts_per_partition = w13_weight.shape[0]
    # TODO: use ep size
    num_experts = num_experts_per_partition * get_tensor_model_parallel_world_size()

    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
        topk_ids, num_experts,
    )

    gateup_input = torch.empty(
        (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_block_quant)
            else hidden_states.dtype
        ),
    )
    if use_fp8_w8a8 and activation_scheme == "dynamic" and not use_block_quant:
        if use_per_token_if_dynamic:
            max_value = torch.max(hidden_states, dim=1).values.to(torch.float32)
            w13_input_scale = max_value / torch.finfo(fp8_dtype).max
        else:
            max_value = (
                torch.max(hidden_states)
                .repeat(num_experts_per_partition)
                .to(torch.float32)
            )
            w13_input_scale = max_value / torch.finfo(fp8_dtype).max

    # PreReorder
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        w13_input_scale,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )
    dispose_tensor(hidden_states)

    if (
        activation_scheme == "dynamic"
        and not use_block_quant
        and use_per_token_if_dynamic
    ):
        scale = torch.empty(
            hidden_states_shape[0] * top_k,
            device=hidden_states_device,
            dtype=torch.float32,
        )
        scale[src2dst] = (
            w13_input_scale.unsqueeze(1)
            .expand(hidden_states_shape[0], top_k)
            .reshape(-1)
        )
        w13_input_scale = scale

    seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        num_experts_per_partition,
        device=hidden_states_device,
        dtype=torch.int64,
    )
    # GroupGemm-0
    gateup_output = grouped_gemm_runner(
        a=gateup_input,
        b=w13_weight,
        c=None,
        c_dtype=hidden_states_dtype,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w13_input_scale,
        # scale_b=(
        #     w13_weight_scale_inv
        #     if use_block_quant
        #     else w13_weight_scale
        # ),
        scale_b=w13_scale,
        block_shape=block_shape,
    )
    del gateup_input

    # Act
    if use_fp8_w8a8 and activation_scheme == "dynamic" and not use_block_quant:
        w2_input_scale = None
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states_dtype,
        )
    else:
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                fp8_dtype
                if (use_fp8_w8a8 and not use_block_quant)
                else hidden_states_dtype
            ),
        )

    if activation == "silu":
        silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            BLOCK_SIZE=512,
        )
    elif activation == "gelu":
        gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            w2_input_scale,
            start_expert_id,
            end_expert_id,
            BLOCK_SIZE=512,
        )
    else:
        raise ValueError(f"Unsupported activation: {activation=}")
    del gateup_output

    if activation_scheme == "dynamic" and not use_block_quant:
        if use_per_token_if_dynamic:
            down_input, w2_input_scale = sglang_per_token_quant_fp8(down_input)
        else:
            w2_input_scale = torch.ones(
                num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states_device,
            )

    # GroupGemm-1
    down_output = torch.empty(
        down_input.shape[0],
        w2_weight.shape[1],
        device=hidden_states_device,
        dtype=hidden_states_dtype,
    )
    down_output = grouped_gemm_runner(
        a=down_input,
        b=w2_weight,
        c=down_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w2_input_scale,
        # scale_b=(
        #     w2_weight_scale_inv
        #     if use_block_quant
        #     else w2_weight_scale
        # ),
        scale_b=w2_scale,
        block_shape=block_shape,
    )
    del down_input

    # PostReorder
    output = torch.empty(
        hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
    )
    post_reorder_triton_kernel[(hidden_states_shape[0],)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states_shape[1],
        0,
        BLOCK_SIZE=512,
    )
    return output
