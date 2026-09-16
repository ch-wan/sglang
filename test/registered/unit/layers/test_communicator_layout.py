"""CPU semantic references, not a distributed runtime or backend qualification."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator_layout import (
    BoundaryLayout,
    GroupPlacement,
)
from sglang.srt.layers.communicator_layout import ParallelAxis as Axis
from sglang.srt.layers.communicator_layout import (
    ParallelGroup,
    Partial,
    Replicate,
    ResidualState,
    Shard,
    TensorLayout,
    TokenPartition,
    TokenSpan,
    validate_layouts,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def partition(ids, padding=0):
    # Fixtures state the joint token assignment directly. Production binding
    # can compress contiguous/strided chunks without enumerating every token.
    return TokenPartition(
        len(ids) + padding, tuple(TokenSpan(t, 1, i) for i, t in enumerate(ids))
    )


def attention_layouts(dp_tokens, cp=1, strategy="contiguous", partial=True):
    """Two TP ranks per (DP, CP) cell; TP is the fastest rank dimension."""
    layouts = []
    for dp, tokens in enumerate(dp_tokens):
        chunks = [
            list(x.tolist())
            for x in torch.tensor(tokens, dtype=torch.int64).tensor_split(cp)
        ]
        if strategy == "interleave":
            chunks = [tokens[c::cp] for c in range(cp)]
        elif strategy == "zigzag":
            halves = [
                list(x.tolist())
                for x in torch.tensor(tokens, dtype=torch.int64).tensor_split(2 * cp)
            ]
            chunks = [halves[c] + halves[2 * cp - c - 1] for c in range(cp)]
        for c in range(cp):
            for tp in range(2):
                rank = (dp * cp + c) * 2 + tp
                dp_group = ParallelGroup(
                    "attn_dp",
                    tuple((d * cp + c) * 2 + tp for d in range(len(dp_tokens))),
                )
                cp_group = ParallelGroup(
                    "attn_cp", tuple((dp * cp + cc) * 2 + tp for cc in range(cp))
                )
                tp_group = ParallelGroup(
                    "attn_tp", ((dp * cp + c) * 2, (dp * cp + c) * 2 + 1)
                )
                layouts.append(
                    TensorLayout(
                        rank,
                        (
                            GroupPlacement(Axis.ATTN_DP, dp_group, Shard()),
                            GroupPlacement(Axis.ATTN_CP, cp_group, Shard()),
                            GroupPlacement(
                                Axis.ATTN_TP,
                                tp_group,
                                Partial() if partial else Replicate(),
                            ),
                        ),
                        partition(chunks[c], padding=1),
                    )
                )
    return layouts


def ffn_layouts(world_size, num_tokens, partial=False):
    group = ParallelGroup("ffn_tp", tuple(range(world_size)))
    return [
        TensorLayout(
            r,
            (
                GroupPlacement(
                    Axis.FFN_TP, group, Partial() if partial else Replicate()
                ),
            ),
            TokenPartition(num_tokens, (TokenSpan(0, num_tokens),)),
        )
        for r in group.ranks
    ]


def local_rows(values, layout):
    # Poison padding: accidentally including it in the logical output fails.
    rows = torch.full(
        (layout.tokens.local_size, values.shape[1]), float("nan"), dtype=values.dtype
    )
    for row, token in enumerate(layout.tokens.token_ids()):
        if token is not None:
            rows[row] = values[token]
    return rows


def allreduce_reference(layouts, local):
    result = dict(local)
    groups = {
        b.group
        for l in layouts
        for b in l.placements
        if isinstance(b.placement, Partial)
    }
    for group in groups:
        summed = torch.stack([local[r] for r in group.ranks]).sum(0)
        for rank in group.ranks:
            result[rank] = summed
    return result


def gather_reference(layouts, local, num_tokens):
    # Explicit gather of already-complete values; never sums replicas.
    values = {}
    for layout in layouts:
        for row, token in enumerate(layout.tokens.token_ids()):
            if token is not None:
                if token in values:
                    torch.testing.assert_close(values[token], local[layout.rank][row])
                values[token] = local[layout.rank][row]
    assert set(values) == set(range(num_tokens))
    width = next(iter(local.values())).shape[1]
    return (
        torch.stack([values[t] for t in range(num_tokens)])
        if num_tokens
        else torch.empty((0, width), dtype=torch.float64)
    )


def rmsnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class TestCommunicatorLayout(unittest.TestCase):
    def _round_trip(self, dp_tokens, cp=1, strategy="contiguous"):
        count = sum(map(len, dp_tokens))
        source = attention_layouts(dp_tokens, cp, strategy)
        replica = attention_layouts(dp_tokens, cp, strategy, partial=False)
        target = ffn_layouts(len(source), count)
        ffn_partial = ffn_layouts(len(source), count, partial=True)
        for layouts in (source, replica, target, ffn_partial):
            validate_layouts(layouts, num_tokens=count)

        x = torch.arange(count * 4, dtype=torch.float64).reshape(count, 4) / 7
        residual = x.flip(1) + 1
        wo = torch.arange(16, dtype=x.dtype).reshape(4, 4) / 17
        local = {}
        for layout in source:
            head = slice((layout.rank % 2) * 2, (layout.rank % 2 + 1) * 2)
            local[layout.rank] = local_rows(x, layout)[:, head] @ wo[head]
        complete = allreduce_reference(source, local)

        # Existing norm-before-gather order: residual remains in its own DP/CP
        # partition while prepared FFN inputs become full on the FFN TP group.
        summed = {l.rank: complete[l.rank] + local_rows(residual, l) for l in replica}
        normalized = {r: rmsnorm(value) for r, value in summed.items()}
        prepared = gather_reference(replica, normalized, count)
        expected_residual = x @ wo + residual
        torch.testing.assert_close(prepared, rmsnorm(expected_residual))
        for hidden, residual_layout in zip(target, replica):
            state = BoundaryLayout(hidden, ResidualState.SEPARATE, residual_layout)
            self.assertEqual(state.residual.tokens, residual_layout.tokens)

        # Dense FFN row-parallel output across a different group. Each rank
        # contributes one distinct intermediate channel; EP is not modeled.
        w1 = torch.arange(4 * len(source), dtype=x.dtype).reshape(4, len(source)) / 31
        w2 = torch.arange(len(source) * 4, dtype=x.dtype).reshape(len(source), 4) / 23
        ffn_values = {
            l.rank: torch.relu(prepared @ w1[:, l.rank : l.rank + 1])
            @ w2[l.rank : l.rank + 1]
            for l in ffn_partial
        }
        ffn_complete = allreduce_reference(ffn_partial, ffn_values)
        global_ffn = gather_reference(target, ffn_complete, count)
        back = {
            l.rank: rmsnorm(local_rows(global_ffn, l) + summed[l.rank]) for l in replica
        }
        actual = gather_reference(replica, back, count)
        expected = rmsnorm(
            torch.relu(rmsnorm(expected_residual) @ w1) @ w2 + expected_residual
        )
        torch.testing.assert_close(actual, expected)

    def test_tp_attention_ffn_boundaries(self):
        self._round_trip([list(range(5))])

    def test_existing_dp_partial_gather_matches_token_reference(self):
        from sglang.srt.layers import dp_attention

        layouts = attention_layouts([[0, 1, 2], [3]])
        validate_layouts(layouts, num_tokens=4)
        values = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        contributions = []
        for layout in layouts:
            dp_rank, tp_rank = divmod(layout.rank, 2)
            start, length = ((0, 3), (3, 1))[dp_rank]
            local = (values[start : start + length] * (tp_rank + 1)).contiguous()
            output = torch.empty_like(values)
            # Run the real buffer placement. The sole collective is captured
            # as each rank's contribution and summed by the CPU reference.
            with (
                patch.object(dp_attention, "memcpy_func", dp_attention.memcpy_cpu),
                patch.object(
                    dp_attention, "get_dp_local_info", return_value=(start, length)
                ),
                patch.object(
                    dp_attention, "world_dp_gather_enabled", return_value=False
                ),
                patch.object(
                    dp_attention,
                    "tensor_model_parallel_all_reduce",
                    side_effect=lambda x: x.clone(),
                ),
            ):
                dp_attention._dp_gather_via_all_reduce(
                    output, local, None, is_partial=True
                )
            contributions.append(output)
        torch.testing.assert_close(torch.stack(contributions).sum(0), values * 3)

    def test_dp2_tp2_to_ffn_tp4_ragged_and_empty(self):
        for tokens in ([list(range(3)), [3]], [[], list(range(4))], [[], []]):
            with self.subTest(tokens=tokens):
                self._round_trip(tokens)

    def test_hybrid_dp2_tp2_cp2_to_ffn_tp8(self):
        for strategy in ("contiguous", "interleave", "zigzag"):
            with self.subTest(strategy=strategy):
                self._round_trip([list(range(6)), [6, 7]], cp=2, strategy=strategy)

    def test_consecutive_attention(self):
        source = attention_layouts([[0, 1, 2]], partial=True)
        target = attention_layouts([[0, 1, 2]], partial=False)
        residual = torch.arange(12, dtype=torch.float64).reshape(3, 4) + 1
        for stage in range(2):
            validate_layouts(source, num_tokens=3)
            validate_layouts(target, num_tokens=3)
            local = {l.rank: local_rows(residual * (l.rank + 1), l) for l in source}
            output = allreduce_reference(source, local)
            output = {
                l.rank: rmsnorm(output[l.rank] + local_rows(residual, l))
                for l in target
            }
            actual = gather_reference(target, output, 3)
            torch.testing.assert_close(actual, rmsnorm(4 * residual))
            residual = actual

    def test_compact_joint_partition_and_padding(self):
        spans = TokenPartition(
            6, (TokenSpan(0, 2, stride=2), TokenSpan(7, 2, local_start=3))
        )
        self.assertEqual(spans.token_ids(), (0, 2, None, 7, 8, None))
        contiguous = attention_layouts([list(range(8))], cp=2)
        interleave = attention_layouts([list(range(8))], cp=2, strategy="interleave")
        self.assertEqual(
            contiguous[0].tokens.local_size, interleave[0].tokens.local_size
        )
        self.assertNotEqual(
            contiguous[0].tokens.token_ids(), interleave[0].tokens.token_ids()
        )
        self.assertEqual(
            replace(
                contiguous[0], placements=tuple(reversed(contiguous[0].placements))
            ),
            contiguous[0],
        )

    def test_group_identity_order_and_live_rebinding(self):
        old = SimpleNamespace(unique_name="tp:0", ranks=[4, 5])
        new = SimpleNamespace(unique_name="tp:1", ranks=[4, 5])
        with get_parallel().override(attn_tp_group=old):
            first = ParallelGroup.from_coordinator(get_parallel().attn_tp_group)
            with get_parallel().override(attn_tp_group=new):
                second = ParallelGroup.from_coordinator(get_parallel().attn_tp_group)
            self.assertEqual(
                ParallelGroup.from_coordinator(get_parallel().attn_tp_group), first
            )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, ParallelGroup("tp:0", (5, 4)))
        self.assertNotEqual(first, ParallelGroup("tp:0", (4, 6)))

    def test_real_zigzag_metadata_matches_joint_partition(self):
        from sglang.srt.layers.cp import padding, zigzag

        strategy = zigzag.ZigzagCPStrategy(cp_size=2)
        expected = ((0, 1, 8, 9, 6, 7, 12), (2, 3, 10, 4, 5, 11))
        values = torch.arange(13).reshape(13, 1)
        for cp_rank in range(2):
            with (
                get_parallel().override(attn_cp_rank=cp_rank),
                patch.object(
                    zigzag, "get_device", return_value=SimpleNamespace(device="cpu")
                ),
            ):
                metadata = strategy.build_metadata(13, [8, 5])
                offsets = [0]
                for length in metadata.split_list:
                    offsets.append(offsets[-1] + length)
                spans, local_start = [], 0
                for index in metadata.zigzag_index:
                    length = metadata.split_list[index]
                    spans.append(TokenSpan(offsets[index], length, local_start))
                    local_start += length
                for padded in (False, True):
                    if padded:
                        with patch.object(
                            padding, "get_cp_padding_align_size", return_value=4
                        ):
                            padding.pad_logical_token_to_physical(metadata)
                    size = metadata.per_rank_actual_token[cp_rank]
                    tokens = TokenPartition(size, tuple(spans))
                    self.assertEqual(
                        tokens.token_ids(),
                        expected[cp_rank] + (None,) * (size - len(expected[cp_rank])),
                    )
                    sharded = strategy.shard_hidden_states(
                        values, SimpleNamespace(attn_cp_metadata=metadata)
                    )
                    self.assertEqual(sharded.shape[0], tokens.local_size)
                    for row, token in enumerate(tokens.token_ids()):
                        if token is not None:
                            self.assertEqual(sharded[row].item(), token)

    def test_reject_group_mismatch_and_missing_peers(self):
        layouts = ffn_layouts(2, 2)
        for group in (
            ParallelGroup("ffn_tp", (1, 0)),
            ParallelGroup("ffn_tp:1", (0, 1)),
            ParallelGroup("ffn_tp", (0, 2)),
        ):
            with (
                self.subTest(group=group),
                self.assertRaisesRegex(ValueError, "Every group member"),
            ):
                validate_layouts(
                    [
                        replace(
                            layouts[0],
                            placements=(
                                GroupPlacement(Axis.FFN_TP, group, Replicate()),
                            ),
                        ),
                        layouts[1],
                    ],
                    num_tokens=2,
                )

    def test_reject_repeated_alias_reduction(self):
        group = ParallelGroup("tp", (0, 1))
        with self.assertRaisesRegex(ValueError, "aliases must not reduce twice"):
            TensorLayout(
                0,
                (
                    GroupPlacement(Axis.ATTN_TP, group, Partial()),
                    GroupPlacement(Axis.FFN_TP, group, Partial()),
                ),
                partition([0]),
            )

    def test_reject_dp_shard_with_cross_dp_replica(self):
        layouts = attention_layouts([[0, 1], [2]], partial=False)
        full = ParallelGroup("ffn_tp", (0, 1, 2, 3))
        layouts = [
            replace(
                l,
                placements=(
                    *l.placements,
                    GroupPlacement(Axis.FFN_TP, full, Replicate()),
                ),
            )
            for l in layouts
        ]
        with self.assertRaisesRegex(ValueError, "align tokens and padding"):
            validate_layouts(layouts, num_tokens=3)

    def test_reject_replicated_partial_contributions(self):
        layouts = ffn_layouts(4, 1, partial=True)
        layouts = [
            replace(
                l,
                placements=(
                    *l.placements,
                    GroupPlacement(
                        Axis.ATTN_TP,
                        ParallelGroup(
                            "attn_tp", (l.rank // 2 * 2, l.rank // 2 * 2 + 1)
                        ),
                        Replicate(),
                    ),
                ),
            )
            for l in layouts
        ]
        with self.assertRaisesRegex(ValueError, "also be declared replicas"):
            validate_layouts(layouts, num_tokens=1)

    def test_reject_partial_order_and_shard_overlap(self):
        layouts = attention_layouts([[0, 1]])
        with self.assertRaisesRegex(ValueError, "align tokens and padding"):
            validate_layouts(
                [layouts[0], replace(layouts[1], tokens=partition([1, 0], padding=1))],
                num_tokens=2,
            )
        layouts = attention_layouts([[0, 1], [1, 2]], partial=False)
        with self.assertRaisesRegex(ValueError, "overlapping tokens"):
            validate_layouts(layouts, num_tokens=3)

    def test_reject_missing_out_of_range_and_undeclared_replicas(self):
        for layouts, count, message in (
            ([TensorLayout(0, (), partition([0]))], 2, "complete token"),
            ([TensorLayout(0, (), partition([2]))], 2, "exceeds"),
            (
                [
                    TensorLayout(0, (), partition([0])),
                    TensorLayout(1, (), partition([0])),
                ],
                1,
                "explicit Replicate",
            ),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_layouts(layouts, num_tokens=count)

    def test_reject_invalid_spans_and_residual_states(self):
        for spans in (
            (TokenSpan(0, 2), TokenSpan(2, 1, local_start=1)),
            (TokenSpan(0, 3),),
        ):
            with self.assertRaises(ValueError):
                TokenPartition(2, spans)
        with self.assertRaisesRegex(ValueError, "same logical token twice"):
            partition([0, 0]).token_ids()
        layout = TensorLayout(0, (), partition([0]))
        for state in (ResidualState.ABSENT, ResidualState.MERGED):
            self.assertIsNone(BoundaryLayout(layout, state).residual)
            with self.assertRaises(ValueError):
                BoundaryLayout(layout, state, layout)
        with self.assertRaises(ValueError):
            BoundaryLayout(layout, ResidualState.SEPARATE)
        with self.assertRaises(ValueError):
            BoundaryLayout(layout, ResidualState.SEPARATE, replace(layout, rank=1))


if __name__ == "__main__":
    unittest.main()
