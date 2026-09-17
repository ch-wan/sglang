# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 MoE's output contract, after expert contributions have been combined.

Expert combine belongs here, inside the MoE wrapper. Preparation only receives
complete values or a pure TP partial. This is not an expert tensor layout or a
new tensor return type, and unadapted backends keep their existing entry points.
"""

from dataclasses import dataclass
from enum import Enum, auto

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.communicator_binding import (
    DenseLayerBoundaryRules,
    FFNExecution,
    FFNReduction,
)
from sglang.srt.layers.communicator_layout import (
    BoundaryLayout,
    GroupPlacement,
    ParallelAxis,
    ParallelGroup,
    ResidualState,
    Shard,
    TensorLayout,
    TokenPartition,
    TokenSpan,
)
from sglang.srt.layers.dp_attention import (
    dp_reduce_scatter_tensor,
    get_dp_global_num_tokens,
    get_local_dp_buffer,
)
from sglang.srt.runtime_context import get_forward, get_parallel


class MoEOutput(Enum):
    GLOBAL = auto()
    ATTENTION_LOCAL = auto()
    SCATTERED = auto()


class MoECombine(Enum):
    NONE = auto()
    REDUCE_SCATTER = auto()
    REDUCE_SCATTERV = auto()


@dataclass(frozen=True)
class MoEBoundaryPlan:
    """One producer decision shared by its wrapper and the decoder consumer.

    ``ffn`` describes reduction left *outside* MoE. ``combine`` is internal
    expert work and must finish before the returned tensor has ``output``.
    """

    ffn: FFNExecution
    combine: MoECombine
    output: MoEOutput

    @classmethod
    def select(cls, *, ep_size, scattered, fuse_next, reduce_scatter, reduce_scatterv):
        complete = FFNExecution()
        if scattered:
            return cls(complete, MoECombine.NONE, MoEOutput.SCATTERED)
        if ep_size > 1:
            # EP contributions never cross the MoE/prepare boundary, including
            # the old pure-EP fused all-reduce + next layernorm optimization.
            if reduce_scatterv:
                return cls(
                    complete, MoECombine.REDUCE_SCATTERV, MoEOutput.ATTENTION_LOCAL
                )
            if reduce_scatter:
                return cls(
                    complete, MoECombine.REDUCE_SCATTER, MoEOutput.ATTENTION_LOCAL
                )
            return cls(complete, MoECombine.NONE, MoEOutput.GLOBAL)
        if fuse_next and reduce_scatter:
            raise ValueError("A TP sum cannot be assigned to two consumers")
        reduction = (
            FFNReduction.NEXT_PREPARE
            if fuse_next
            else FFNReduction.POSTPROCESS
            if reduce_scatter
            else FFNReduction.COMPUTE
        )
        return cls(FFNExecution(reduction), MoECombine.NONE, MoEOutput.GLOBAL)

    def scope(self):
        # Existing expert implementations read these flags. For internal DP
        # combine, suppress their all-reduces until finish() runs inside MoE.
        return get_forward().scoped(
            fuse_mlp_allreduce=self.ffn.reduction is FFNReduction.NEXT_PREPARE,
            mlp_reduce_scatter=(
                self.ffn.reduction is FFNReduction.POSTPROCESS
                or self.combine is not MoECombine.NONE
            ),
        )

    def finish(self, hidden_states):
        if self.combine is MoECombine.NONE:
            return hidden_states
        parallel = get_parallel()
        group = (
            get_tp_group()
            if parallel.tp_size == parallel.attn_dp_size
            else parallel.attn_tp_group
        )
        local = get_local_dp_buffer(group)
        if self.combine is MoECombine.REDUCE_SCATTERV:
            get_tp_group().reduce_scatterv(
                hidden_states, output=local, sizes=get_dp_global_num_tokens()
            )
        else:
            dp_reduce_scatter_tensor(local, hidden_states)
        return local

    def bind_output(self, forward_batch, *, valid_tokens, physical_tokens=None):
        """Host-only description of the returned tensor and residual pair."""
        parallel = get_parallel()
        if parallel.moe_dp_size != 1 or parallel.dwdp_size > 1:
            raise NotImplementedError("MoE DP / DWDP requires its own output adapter")
        if parallel.moe_ep_size > 1 and self.ffn.reduction is not FFNReduction.COMPUTE:
            raise ValueError("An EP contribution cannot be described as FFN TP partial")
        if self.combine is MoECombine.REDUCE_SCATTERV and (
            parallel.tp_size != parallel.attn_dp_size or parallel.moe_tp_size != 1
        ):
            raise ValueError("Expert reduce-scatterv requires one rank per DP shard")
        if self.output is MoEOutput.SCATTERED and parallel.moe_tp_size != 1:
            raise NotImplementedError(
                "A2A with tensor-parallel experts needs its own adapter"
            )
        binding = DenseLayerBoundaryRules(False, True).bind(
            forward_batch,
            valid_tokens=valid_tokens,
            physical_tokens=physical_tokens,
            execution=self.ffn,
        )
        if self.output is MoEOutput.GLOBAL:
            return binding.ffn_output
        if self.output is MoEOutput.ATTENTION_LOCAL:
            return binding.prepare_attn_target
        local = binding.prepare_attn_target.hidden.tokens
        width = parallel.attn_tp_size
        if local.local_size % width:
            raise ValueError("Scattered MoE input requires TP-aligned token capacity")
        size = local.local_size // width
        start = parallel.attn_tp_rank * size
        spans = []
        for span in local.spans:
            lo = max(start, span.local_start)
            hi = min(start + size, span.local_start + span.length)
            if lo < hi:
                spans.append(
                    TokenSpan(
                        span.token_start + lo - span.local_start, hi - lo, lo - start
                    )
                )
        layout = TensorLayout(
            parallel.world_rank,
            (
                GroupPlacement(
                    ParallelAxis.FFN_DP,
                    ParallelGroup.from_coordinator(parallel.tp_group),
                    Shard(),
                ),
            ),
            TokenPartition(size, tuple(spans)),
        )
        return BoundaryLayout(layout, ResidualState.SEPARATE, layout)
