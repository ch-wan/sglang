"""Host binding and producer-state lifetime, independent of GPU collectives."""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator_binding import (
    DenseLayerBoundaryRules,
    FFNExecution,
    FFNReduction,
)
from sglang.srt.layers.communicator_layout import Partial, validate_layouts
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass
class Padding:
    max_len: bool

    def is_max_len(self):
        return self.max_len


def batch(counts=None, max_len=False, rows=4):
    return SimpleNamespace(
        global_num_tokens_cpu=counts,
        dp_padding_mode=Padding(max_len) if counts is not None else None,
        input_ids=torch.empty(rows, dtype=torch.int64),
    )


def topology(rank=0, dp=1, tp=2, generation=0):
    # Nonzero PP offset and non-contiguous global ranks expose accidental use
    # of rank modulo TP instead of the current coordinator's ordered members.
    ranks = [10 + 2 * r for r in range(dp * tp)]
    d, t = divmod(rank, tp)
    return get_parallel().override(
        world_rank=ranks[rank],
        tp_size=dp * tp,
        tp_rank=rank,
        attn_cp_size=1,
        attn_dp_size=dp,
        attn_dp_rank=d,
        attn_tp_size=tp,
        attn_tp_rank=t,
        tp_group=SimpleNamespace(unique_name=f"tp:{generation}", ranks=ranks),
        attn_tp_group=SimpleNamespace(
            unique_name=f"attn_tp:{d}:{generation}", ranks=ranks[d * tp : (d + 1) * tp]
        ),
    )


RULES = DenseLayerBoundaryRules(attention_reduce_results=False, ffn_reduce_results=True)


def partial(layout):
    return any(isinstance(p.placement, Partial) for p in layout.placements)


class TestDenseBoundaryBinding(unittest.TestCase):
    def bind_all(self, valid, physical, max_len=False, execution=FFNExecution(), tp=2):
        bindings = []
        for rank in range(len(valid) * tp):
            with topology(rank, len(valid), tp):
                bindings.append(
                    RULES.bind(
                        batch(physical, max_len),
                        valid_tokens=valid,
                        execution=execution,
                    )
                )
        for field in (
            "prepare_ffn_source",
            "prepare_ffn_target",
            "ffn_output",
            "prepare_attn_source",
            "prepare_attn_target",
        ):
            for member in ("hidden", "residual"):
                validate_layouts(
                    [getattr(getattr(b, field), member) for b in bindings],
                    num_tokens=sum(valid),
                )
        return bindings

    def test_live_counts_and_padding(self):
        for valid, physical, max_len in (
            ([3, 1], [4, 2], False),
            ([1, 3], [4, 4], True),
            ([0, 3], [0, 4], False),
            ([0, 3], [4, 4], True),
            ([0, 0], [0, 0], False),
        ):
            with self.subTest(valid=valid, max_len=max_len):
                bindings = self.bind_all(valid, physical, max_len)
                expected = []
                offset = 0
                for v, p in zip(valid, physical):
                    expected.extend(range(offset, offset + v))
                    expected.extend([None] * (p - v))
                    offset += v
                self.assertEqual(
                    bindings[0].ffn_output.hidden.tokens.token_ids(), tuple(expected)
                )
                for r, b in enumerate(bindings):
                    d = r // 2
                    start = sum(physical[:d])
                    self.assertEqual(
                        b.prepare_attn_source.hidden.tokens.token_ids(),
                        tuple(expected[start : start + physical[d]]),
                    )
                    self.assertTrue(partial(b.prepare_ffn_source.hidden))
                    self.assertFalse(partial(b.prepare_ffn_source.residual))
                    self.assertFalse(partial(b.ffn_output.hidden))

    def test_tp1_and_no_dp_metadata(self):
        self.bind_all([3], [4], tp=1)
        with topology(tp=1):
            b = RULES.bind(batch(rows=5), valid_tokens=[3], execution=FFNExecution())
        self.assertFalse(partial(b.prepare_ffn_source.hidden))
        self.assertEqual(b.ffn_output.hidden.tokens.token_ids(), (0, 1, 2, None, None))

    def test_explicit_graph_capacity_and_snapshot(self):
        counts = [3, 1]
        forward = batch(counts, max_len=True)
        with topology(dp=2):
            b = RULES.bind(
                forward,
                valid_tokens=counts,
                physical_tokens=[8, 8],
                execution=FFNExecution(),
            )
            counts[:] = [1, 2]
            next_b = RULES.bind(
                forward,
                valid_tokens=counts,
                physical_tokens=[8, 8],
                execution=FFNExecution(),
            )
        self.assertEqual(b.ffn_output.hidden.tokens.token_ids()[8], 3)
        self.assertEqual(next_b.ffn_output.hidden.tokens.token_ids()[8], 1)
        self.assertEqual(b.ffn_output.hidden.tokens.local_size, 16)

    def test_producer_survives_nested_scopes_and_exception(self):
        before = FFNExecution.capture()
        with FFNExecution(FFNReduction.NEXT_PREPARE).scope():
            producer = FFNExecution.capture()
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with FFNExecution(FFNReduction.POSTPROCESS).scope():
                    other_producer = FFNExecution.capture()
                    raise RuntimeError("abort")
            self.assertEqual(FFNExecution.capture(), producer)
        self.assertEqual(FFNExecution.capture(), before)
        with topology():
            b = RULES.bind(batch(), valid_tokens=[4], execution=producer)
        self.assertTrue(partial(b.prepare_attn_source.hidden))
        self.assertFalse(partial(b.prepare_attn_source.residual))
        self.assertFalse(partial(b.prepare_attn_target.hidden))
        self.assertEqual(other_producer.reduction, FFNReduction.POSTPROCESS)

    def test_postprocess_completes_sum_before_next_prepare(self):
        b = self.bind_all([3, 1], [4, 4], True, FFNExecution(FFNReduction.POSTPROCESS))[
            0
        ]
        self.assertTrue(partial(b.ffn_output.hidden))
        self.assertFalse(partial(b.prepare_attn_source.hidden))
        self.assertEqual(b.prepare_attn_source, b.prepare_attn_target)
        self.assertEqual(b.prepare_attn_source.hidden.tokens.local_size, 4)
        self.assertEqual(b.ffn_output.hidden.tokens.local_size, 8)

    def test_group_rebind_does_not_reuse_handles_or_ids(self):
        with topology(generation=0):
            first = RULES.bind(batch(), valid_tokens=[4], execution=FFNExecution())
        with topology(generation=1):
            second = RULES.bind(batch(), valid_tokens=[4], execution=FFNExecution())
        self.assertNotEqual(first.ffn_output.hidden, second.ffn_output.hidden)
        self.assertEqual(
            first.ffn_output.hidden.tokens, second.ffn_output.hidden.tokens
        )

    def test_reject_ambiguous_or_unsupported_execution(self):
        with get_forward().scoped(fuse_mlp_allreduce=True, mlp_reduce_scatter=True):
            with self.assertRaisesRegex(ValueError, "two consumers"):
                FFNExecution.capture()
        for flags in ({"sp_active": True}, {"attn_input_scattered": True}):
            with topology(), get_forward().scoped(**flags):
                with self.assertRaises(NotImplementedError):
                    RULES.bind(batch(), valid_tokens=[4], execution=FFNExecution())
        for decision, max_len in (
            (FFNReduction.NEXT_PREPARE, True),
            (FFNReduction.POSTPROCESS, False),
        ):
            with topology(dp=2), self.assertRaises(NotImplementedError):
                RULES.bind(
                    batch([4, 4], max_len),
                    valid_tokens=[4, 4],
                    execution=FFNExecution(decision),
                )
        with topology(), get_parallel().override(attn_cp_size=2):
            with self.assertRaises(NotImplementedError):
                RULES.bind(batch(), valid_tokens=[4], execution=FFNExecution())

    def test_reject_wrong_counts_and_rank_order(self):
        with topology(dp=2):
            for valid, physical, max_len in (
                ([3], [4, 4], True),
                ([5, 1], [4, 4], True),
                ([1, 1], [4, 2], True),
                ([-1, 1], [4, 4], True),
            ):
                with (
                    self.subTest(valid=valid, physical=physical),
                    self.assertRaises(ValueError),
                ):
                    RULES.bind(
                        batch(physical, max_len),
                        valid_tokens=valid,
                        execution=FFNExecution(),
                    )
            with get_parallel().override(
                attn_tp_group=SimpleNamespace(unique_name="bad", ranks=[12, 10])
            ):
                with self.assertRaisesRegex(ValueError, "rank ordering"):
                    RULES.bind(
                        batch([4, 4]), valid_tokens=[4, 4], execution=FFNExecution()
                    )

    def test_actual_dp_scatter_matches_bound_local_rows(self):
        from sglang.srt.layers import dp_attention

        for valid, physical, max_len in (
            ([3, 1], [4, 2], False),
            ([0, 3], [4, 4], True),
        ):
            bindings = self.bind_all(valid, physical, max_len)
            # Fill physical rows distinctly, including padding: compare what
            # the existing copy path selects, not a reimplemented collective.
            full = torch.arange(sum(physical), dtype=torch.float32).reshape(-1, 1)
            for d in range(2):
                local = torch.empty(physical[d], 1)
                start = sum(physical[:d])
                with (
                    patch.object(
                        dp_attention,
                        "get_dp_local_info",
                        return_value=(start, physical[d]),
                    ),
                    patch.object(dp_attention, "memcpy_func", dp_attention.memcpy_cpu),
                ):
                    dp_attention.dp_scatter(local, full, batch(physical, max_len))
                tokens = bindings[d * 2].prepare_attn_source.hidden.tokens.token_ids()
                full_tokens = bindings[d * 2].ffn_output.hidden.tokens.token_ids()
                self.assertEqual(
                    local.flatten().tolist(), list(range(start, start + physical[d]))
                )
                self.assertEqual(tokens, full_tokens[start : start + physical[d]])

    def test_qwen3_constructed_projections_and_adapter(self):
        from transformers import Qwen3Config

        from sglang.srt.models import qwen3

        for dp, tp in ((1, 1), (1, 2), (2, 2)):
            with (
                get_context().override_server_args(
                    tp_size=dp * tp, dp_size=dp, enable_dp_attention=dp > 1
                ),
                topology(dp=dp, tp=tp),
                get_parallel().override(
                    attn_cp_rank=0, moe_dp_size=1, moe_tp_size=dp * tp, moe_ep_size=1
                ),
                # The CPU environment lacks vLLM's rotary kernel. Rope is unrelated
                # to projection construction or the preparation boundary contract.
                patch.object(qwen3, "get_rope", return_value=torch.nn.Identity()),
                patch("sglang.srt.layers.communicator.get_moe_cp_size", return_value=1),
            ):
                layer = qwen3.Qwen3DecoderLayer(
                    Qwen3Config(
                        hidden_size=32,
                        intermediate_size=64,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        num_hidden_layers=2,
                        head_dim=8,
                    )
                )
                forward = batch([4] * dp, max_len=True)
                b = layer.bind_boundary_layouts(forward, valid_tokens=[3] * dp)
                self.assertEqual(
                    [
                        t
                        for t in b.ffn_output.hidden.tokens.token_ids()
                        if t is not None
                    ],
                    list(range(3 * dp)),
                )
                self.assertFalse(layer.self_attn.o_proj.reduce_results)
                self.assertTrue(layer.mlp.down_proj.reduce_results)
                with FFNExecution(FFNReduction.NEXT_PREPARE).scope():
                    with self.assertRaisesRegex(
                        ValueError, "no delayed-reduction handoff"
                    ):
                        layer.bind_boundary_layouts(forward, valid_tokens=[3] * dp)
                layer.mlp.down_proj.use_decode_attn_tp = True
                with self.assertRaises(NotImplementedError):
                    layer.bind_boundary_layouts(forward, valid_tokens=[3] * dp)

    def test_compile_flags_no_host_layout_in_graph(self):
        # Same graph-visible flag slots used by the row-parallel skip helper.
        # Host bindings are intentionally created outside the traced function.
        from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce

        compilations = []

        def backend(graph, inputs):
            compilations.append(graph)
            return graph.forward

        def compute(x):
            return x + 1 if should_skip_mlp_all_reduce() else x * 2

        compiled = torch.compile(compute, backend=backend, fullgraph=True, dynamic=True)
        try:
            for decision in (
                FFNReduction.COMPUTE,
                FFNReduction.NEXT_PREPARE,
                FFNReduction.POSTPROCESS,
                FFNReduction.COMPUTE,
            ):
                with FFNExecution(decision).scope():
                    for rows in (3, 7, 5):
                        x = torch.arange(rows, dtype=torch.float32)
                        torch.testing.assert_close(compiled(x), compute(x))
            self.assertEqual(len(compilations), 3)
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
