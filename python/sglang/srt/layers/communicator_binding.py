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
"""Host-side binding for ordinary dense TP/DP preparation boundaries.

Binding is explicit and on demand, outside compiled forwards and CUDA capture.
It neither selects collectives nor stores a current layout on a global context.
A caller must carry the producer's execution decision past its flag scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from sglang.srt.layers.communicator_layout import (
    BoundaryLayout,
    GroupPlacement,
    ParallelAxis,
    ParallelGroup,
    Partial,
    Replicate,
    ResidualState,
    Shard,
    TensorLayout,
    TokenPartition,
    TokenSpan,
)
from sglang.srt.runtime_context import get_forward, get_parallel


class FFNPreparation(Enum):
    LOCAL = auto()
    TP_REDUCE = auto()
    DP_GATHER = auto()


class FFNReduction(Enum):
    COMPUTE = auto()
    POSTPROCESS = auto()
    NEXT_PREPARE = auto()


@dataclass(frozen=True)
class FFNExecution:
    """Where this producer's row-parallel sum is completed.

    Capture inside the producer scope, or use ``scope`` to publish a decision
    already selected by its policy. A downstream consumer receives this value
    explicitly, never re-reads its own flags to reconstruct the producer exit.
    """

    reduction: FFNReduction = FFNReduction.COMPUTE

    def __post_init__(self):
        if not isinstance(self.reduction, FFNReduction):
            raise ValueError("Expected an explicit FFN reduction decision")

    @classmethod
    def capture(cls) -> FFNExecution:
        flags = get_forward()
        if flags.fuse_mlp_allreduce and flags.mlp_reduce_scatter:
            raise ValueError("FFN reduction cannot be assigned to two consumers")
        if flags.fuse_mlp_allreduce:
            return cls(FFNReduction.NEXT_PREPARE)
        if flags.mlp_reduce_scatter:
            return cls(FFNReduction.POSTPROCESS)
        return cls()

    def scope(self):
        return get_forward().scoped(
            fuse_mlp_allreduce=self.reduction is FFNReduction.NEXT_PREPARE,
            mlp_reduce_scatter=self.reduction is FFNReduction.POSTPROCESS,
        )


@dataclass(frozen=True)
class DenseBoundaryBinding:
    """One batch's boundaries, including the postprocess handoff.

    ``prepare_attn_source`` is the actual FFN exit *after* postprocess (or the
    unreduced exit when postprocess is intentionally skipped for fusion).
    These descriptions do not share mutable state with a ForwardBatch.
    """

    execution: FFNExecution
    ffn_preparation: FFNPreparation
    prepare_ffn_source: BoundaryLayout
    prepare_ffn_target: BoundaryLayout
    ffn_output: BoundaryLayout
    prepare_attn_source: BoundaryLayout
    prepare_attn_target: BoundaryLayout


@dataclass(frozen=True)
class DenseLayerBoundaryRules:
    """Static contract of a dense attention/FFN pair, not an alternating graph.

    The ordinary adapter expects O-proj to leave its sum for prepare_ffn and
    down-proj to own its all-reduce unless the execution decision defers it.
    Other producers must provide their own exit rules.
    """

    attention_reduce_results: bool
    ffn_reduce_results: bool

    def _groups(self):
        if self.attention_reduce_results or not self.ffn_reduce_results:
            raise NotImplementedError("Expected deferred O-proj and reducing down-proj")
        parallel = get_parallel()
        if parallel.attn_cp_size != 1:
            raise NotImplementedError("This adapter supports ordinary TP/DP only")

        tp = ParallelGroup.from_coordinator(parallel.tp_group)
        attn_tp = ParallelGroup.from_coordinator(parallel.attn_tp_group)
        rank = parallel.world_rank
        dp_size, tp_size = parallel.attn_dp_size, parallel.attn_tp_size
        if len(tp.ranks) != dp_size * tp_size or len(attn_tp.ranks) != tp_size:
            raise ValueError("Dense TP and attention DP x TP geometry disagree")
        local_rank = tp.ranks.index(rank)
        dp_rank, attn_rank = divmod(local_rank, tp_size)
        if (
            attn_tp.ranks != tp.ranks[dp_rank * tp_size : (dp_rank + 1) * tp_size]
            or dp_rank != parallel.attn_dp_rank
            or attn_rank != parallel.attn_tp_rank
        ):
            raise ValueError("Attention groups do not match the current rank ordering")

        return tp, attn_tp

    def ffn_preparation(self) -> FFNPreparation:
        """Compile the declared ordinary boundary to a supported implementation.

        Group sizes are used only after validating their roles and ordered
        membership. Runtime collectives still resolve live coordinator handles.
        """
        tp, attn_tp = self._groups()
        if len(tp.ranks) > len(attn_tp.ranks):
            return FFNPreparation.DP_GATHER
        if len(attn_tp.ranks) > 1:
            return FFNPreparation.TP_REDUCE
        return FFNPreparation.LOCAL

    def bind(
        self,
        forward_batch,
        *,
        valid_tokens: Sequence[int],
        execution: FFNExecution,
        physical_tokens: Sequence[int] | None = None,
    ) -> DenseBoundaryBinding:
        """Bind a single forward/microbatch; no D2H reads or handle caching.

        Counts are in the *compute* token domain, not necessarily the original
        scheduler batch (speculation can expand it). Physical counts default
        to prepare_dp's padded counts; graph callers supply capture capacities.
        """
        flags = get_forward()
        if flags.sp_active or flags.attn_input_scattered:
            raise NotImplementedError("This adapter supports ordinary TP/DP only")
        tp, attn_tp = self._groups()
        rank = get_parallel().world_rank
        tp_size = len(attn_tp.ranks)
        dp_size = len(tp.ranks) // tp_size
        dp_rank, attn_rank = divmod(tp.ranks.index(rank), tp_size)

        valid = tuple(valid_tokens)
        if physical_tokens is None:
            physical_tokens = forward_batch.global_num_tokens_cpu
            if physical_tokens is None:
                if dp_size != 1:
                    raise ValueError("DP binding requires prepared physical counts")
                physical_tokens = (forward_batch.input_ids.shape[0],)
        physical = tuple(physical_tokens)
        if len(valid) != dp_size or len(physical) != dp_size:
            raise ValueError("Expected one valid/physical count per attention DP rank")
        if any(type(n) is not int for n in (*valid, *physical)):
            raise ValueError("Host binding requires integer token counts")
        if any(v < 0 or p < v for v, p in zip(valid, physical)):
            raise ValueError("Valid tokens must fit in their physical DP slice")
        if dp_size > 1:
            if any(p % tp_size for p in physical):
                raise ValueError("Prepared DP capacities must align to attention TP")
            if forward_batch.dp_padding_mode is None:
                raise ValueError("DP binding must follow prepare_dp")
            if forward_batch.dp_padding_mode.is_max_len() and len(set(physical)) != 1:
                raise ValueError("MAX_LEN requires equal physical DP capacities")
        reduction = execution.reduction
        if reduction is FFNReduction.NEXT_PREPARE and dp_size != 1:
            raise NotImplementedError("Deferred fusion does not support DP attention")
        if reduction is FFNReduction.POSTPROCESS and (
            dp_size == 1 or not forward_batch.dp_padding_mode.is_max_len()
        ):
            raise NotImplementedError("Only MAX_LEN DP reduce-scatter is described")

        # Compact joint token mapping: offsets in the logical domain count only
        # valid tokens; buffer offsets include each DP rank's padding.
        spans = []
        token_start = local_start = 0
        for v, p in zip(valid, physical):
            if v:
                spans.append(TokenSpan(token_start, v, local_start))
            token_start += v
            local_start += p
        full_tokens = TokenPartition(sum(physical), tuple(spans))
        local_tokens = TokenPartition(
            physical[dp_rank],
            (TokenSpan(sum(valid[:dp_rank]), valid[dp_rank]),)
            if valid[dp_rank]
            else (),
        )
        # An ownership group, not an extra process group or collective axis.
        dp = ParallelGroup(
            f"{tp.name}/attn_dp/{attn_rank}", tp.ranks[attn_rank::tp_size]
        )

        def attention(partial=False):
            return TensorLayout(
                rank,
                (
                    *(
                        (GroupPlacement(ParallelAxis.ATTN_DP, dp, Shard()),)
                        if dp_size > 1
                        else ()
                    ),
                    GroupPlacement(
                        ParallelAxis.ATTN_TP,
                        attn_tp,
                        Partial() if partial and tp_size > 1 else Replicate(),
                    ),
                ),
                local_tokens,
            )

        def ffn(partial=False):
            return TensorLayout(
                rank,
                (
                    GroupPlacement(
                        ParallelAxis.FFN_TP,
                        tp,
                        Partial() if partial and len(tp.ranks) > 1 else Replicate(),
                    ),
                ),
                full_tokens,
            )

        residual = attention()

        def boundary(hidden):
            return BoundaryLayout(hidden, ResidualState.SEPARATE, residual)

        ffn_output = boundary(ffn(partial=reduction is not FFNReduction.COMPUTE))
        return DenseBoundaryBinding(
            execution=execution,
            ffn_preparation=self.ffn_preparation(),
            prepare_ffn_source=boundary(attention(partial=True)),
            prepare_ffn_target=boundary(ffn()),
            ffn_output=ffn_output,
            prepare_attn_source=(
                ffn_output
                if reduction is FFNReduction.NEXT_PREPARE
                else boundary(attention())
            ),
            prepare_attn_target=boundary(attention()),
        )
