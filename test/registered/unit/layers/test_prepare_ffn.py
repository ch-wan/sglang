import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator_binding import (
    DenseLayerBoundaryRules,
    FFNExecution,
    FFNPreparation,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

RULES = DenseLayerBoundaryRules(False, True)


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


class Norm(torch.nn.Module):
    def forward(self, x, residual=None):
        if residual is None:
            return rms(x)
        residual.add_(x)
        return rms(residual), residual

    def forward_with_allreduce_fusion(self, x, residual, use_attn_tp_group):
        assert use_attn_tp_group
        # Fresh residual models the FlashInfer alias contract, not its kernel.
        out = x * 2 + residual
        return rms(out), out


@contextmanager
def environment(dp=1, tp=1, rank=0, cp=1):
    dp_rank, tp_rank = divmod(rank, tp)
    with (
        get_context().override_server_args(tp_size=dp * tp * cp, dp_size=dp),
        get_parallel().override(
            world_rank=rank,
            tp_rank=rank,
            tp_size=dp * tp * cp,
            attn_dp_rank=dp_rank,
            attn_dp_size=dp,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp,
            attn_cp_rank=0,
            attn_cp_size=cp,
            tp_group=SimpleNamespace(unique_name="tp", ranks=list(range(dp * tp * cp))),
            attn_tp_group=SimpleNamespace(
                unique_name=f"atp:{dp_rank}",
                ranks=list(range(dp_rank * tp, (dp_rank + 1) * tp)),
            ),
        ),
        patch.object(comm, "get_moe_cp_size", return_value=1),
        patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False),
        patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=False),
        patch.object(
            comm, "use_symmetric_memory", side_effect=lambda *a, **kw: nullcontext()
        ),
        patch.object(comm, "get_tp_group", return_value=None),
    ):
        yield


def communicator(
    *, rules=RULES, force_norm=False, cp=False, cls=comm.LayerCommunicator
):
    modes = comm.LayerScatterModes(
        comm.ScatterMode.TP_ATTN_FULL,
        comm.ScatterMode.TP_ATTN_FULL,
        comm.ScatterMode.MOE_FULL if cp else comm.ScatterMode.FULL,
        comm.ScatterMode.TP_ATTN_FULL,
        comm.ScatterMode.TP_ATTN_FULL,
    )
    return cls(
        modes,
        Norm(),
        Norm(),
        boundary_rules=rules,
        force_layernorm_before_dp_gather=force_norm,
    )


def batch(counts, mode=DpPaddingMode.SUM_LEN, forward_mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        global_num_tokens_cpu=counts, dp_padding_mode=mode, forward_mode=forward_mode
    )


class TestPrepareFFN(unittest.TestCase):
    def test_rule_selection_matches_bound_layout(self):
        for dp, tp, expected in (
            (1, 1, FFNPreparation.LOCAL),
            (1, 2, FFNPreparation.TP_REDUCE),
            (2, 1, FFNPreparation.DP_GATHER),
            (2, 2, FFNPreparation.DP_GATHER),
        ):
            with self.subTest(dp=dp, tp=tp), environment(dp, tp):
                layer = communicator()
                binding = RULES.bind(
                    batch([4] * dp), valid_tokens=[3] * dp, execution=FFNExecution()
                )
                self.assertEqual(layer._ffn_preparation, expected)
                self.assertEqual(layer._ffn_preparation, binding.ffn_preparation)
                self.assertIsNotNone(layer._prepare_ffn_fn)

    def test_tp_math_alias_and_aux_snapshot(self):
        for tp in (1, 2):
            for fused in (False, True):
                with self.subTest(tp=tp, fused=fused), environment(tp=tp):
                    layer = communicator()
                    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
                    residual = torch.full_like(x, 2)
                    original = residual.clone()
                    captured = []
                    # Observe capture ownership without running the preceding
                    # input norm; prepare_mlp and the snapshot logic are real.
                    layer.prepare_attn = lambda h, r, f, **kw: (h, r)
                    with (
                        patch.object(
                            comm,
                            "apply_flashinfer_allreduce_fusion",
                            return_value=fused,
                        ),
                        patch.object(
                            comm,
                            "attention_tensor_model_parallel_all_reduce",
                            side_effect=lambda t: t * tp,
                        ) as ar,
                    ):
                        layer.prepare_attn_and_capture_last_layer_outputs(
                            x, residual, batch([3]), captured
                        )
                        actual, out_residual = layer.prepare_mlp(
                            x, residual, batch([3])
                        )
                    torch.testing.assert_close(actual, rms(x * tp + original))
                    torch.testing.assert_close(out_residual, x * tp + original)
                    torch.testing.assert_close(captured[0], original)
                    if fused and tp > 1:
                        self.assertNotEqual(
                            out_residual.data_ptr(), residual.data_ptr()
                        )
                        self.assertEqual(captured[0].data_ptr(), residual.data_ptr())
                        ar.assert_not_called()
                    else:
                        self.assertEqual(out_residual.data_ptr(), residual.data_ptr())
                        self.assertNotEqual(captured[0].data_ptr(), residual.data_ptr())
                        self.assertEqual(ar.call_count, int(tp > 1))

    def test_dp_gather_order_residual_and_empty_ranks(self):
        for tp, force, counts, mode in (
            (1, False, [3, 1], DpPaddingMode.SUM_LEN),
            (2, False, [4, 2], DpPaddingMode.SUM_LEN),
            (2, True, [4, 4], DpPaddingMode.MAX_LEN),
            (2, False, [0, 4], DpPaddingMode.SUM_LEN),
            (2, True, [0, 4], DpPaddingMode.SUM_LEN),
            (2, False, [0, 0], DpPaddingMode.SUM_LEN),
        ):
            values = (
                torch.arange(sum(counts) * 4, dtype=torch.float32).reshape(-1, 4) / 10
            )
            residuals = values + 2
            total = values * sum(range(1, tp + 1)) + residuals
            before = force or tp == 1
            for rank in range(2 * tp):
                d, t = divmod(rank, tp)
                start, size = sum(counts[:d]), counts[d]
                sl = slice(start, start + size)
                x, residual = values[sl] * (t + 1), residuals[sl].clone()
                events = []

                def gather(out, local, forward):
                    expected_local = (
                        rms(total[sl])
                        if before
                        else values[sl] * (t + 1) + (residuals[sl] if t == 0 else 0)
                    )
                    torch.testing.assert_close(local, expected_local)
                    out.copy_(rms(total) if before else total)
                    events.append("gather")

                def scatter(out, full, forward):
                    out.copy_(full[sl])
                    events.append("scatter")

                with (
                    self.subTest(tp=tp, force=force, counts=counts, rank=rank),
                    environment(2, tp, rank),
                    ExitStack() as stack,
                ):
                    layer = communicator(force_norm=force)
                    stack.enter_context(
                        patch.object(
                            comm,
                            "attention_tensor_model_parallel_all_reduce",
                            side_effect=lambda h: values[sl] * sum(range(1, tp + 1)),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            comm,
                            "get_global_dp_buffer",
                            side_effect=lambda g: torch.empty_like(total),
                        )
                    )
                    gather_partial = stack.enter_context(
                        patch.object(comm, "dp_gather_partial", side_effect=gather)
                    )
                    gather_replica = stack.enter_context(
                        patch.object(comm, "dp_gather_replicate", side_effect=gather)
                    )
                    stack.enter_context(
                        patch.object(comm, "dp_scatter", side_effect=scatter)
                    )
                    output, new_residual = layer.prepare_mlp(
                        x, residual, batch(counts, mode)
                    )
                    torch.testing.assert_close(output, rms(total))
                    torch.testing.assert_close(new_residual, total[sl])
                    self.assertEqual(new_residual.data_ptr(), residual.data_ptr())
                    self.assertEqual(
                        events, ["gather"] if before else ["gather", "scatter"]
                    )
                    self.assertEqual(gather_replica.call_count, int(before))
                    self.assertEqual(gather_partial.call_count, int(not before))

    def test_quant_communication_and_npu_cache_forwarding(self):
        with (
            environment(tp=2),
            get_context().override_server_args(enable_quant_communications=True),
        ):
            layer = communicator()
            x, residual = torch.ones(2, 4), torch.ones(2, 4)
            cache = [object(), object()]
            events = []

            def reduce(x):
                events.append("quant_ar")
                return x * 2

            def prefetch(h, c):
                self.assertIs(c, cache)
                torch.testing.assert_close(h, x * 2)
                events.append("cache")

            with (
                patch.object(
                    comm,
                    "attention_tensor_model_parallel_quant_all_reduce",
                    side_effect=reduce,
                ),
                patch.object(
                    comm, "attention_tensor_model_parallel_all_reduce"
                ) as plain,
                patch.object(comm, "_is_npu", True),
                patch.object(
                    comm, "prepare_weight_cache", side_effect=prefetch, create=True
                ),
            ):
                layer.prepare_mlp(x, residual, batch([2]), cache=cache)
                plain.assert_not_called()
            self.assertEqual(events, ["quant_ar", "cache"])

    def test_special_paths_keep_legacy_entry(self):
        with environment():
            layer = communicator()
            sentinel = (object(), object())
            legacy = Mock(return_value=sentinel)
            layer._communicate_with_all_reduce_and_layer_norm_fn = legacy
            layer._prepare_ffn_fn = Mock(
                side_effect=AssertionError("ordinary path entered")
            )
            with get_forward().scoped(attn_input_scattered=True):
                self.assertIs(layer.prepare_mlp(None, None, None), sentinel)
            legacy.assert_called_once()
            layer._sp_variant = SimpleNamespace(prepare_mlp=Mock(return_value=sentinel))
            with get_forward().scoped(sp_active=True):
                self.assertIs(layer.prepare_mlp(None, None, None), sentinel)
            layer._sp_variant.prepare_mlp.assert_called_once()
            self.assertIsNone(communicator(rules=None)._prepare_ffn_fn)

            class Specialized(comm.LayerCommunicator):
                pass

            self.assertIsNone(communicator(cls=Specialized)._prepare_ffn_fn)
        with environment(dp=2, cp=2):
            self.assertIsNone(communicator(cp=True)._prepare_ffn_fn)

    def test_local_fullgraph_dynamic_shapes(self):
        with environment():
            layer = communicator()
            fwd = batch([4])
            compilations = []

            def backend(graph, inputs):
                compilations.append(graph)
                return graph.forward

            def run(x, residual):
                return layer.prepare_mlp(x, residual, fwd)

            compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
            try:
                for rows in (3, 7, 5):
                    x, residual = torch.randn(rows, 4), torch.randn(rows, 4)
                    expected = x + residual
                    y, r = compiled(x, residual)
                    torch.testing.assert_close(y, rms(expected))
                    torch.testing.assert_close(r, expected)
                    self.assertEqual(r.data_ptr(), residual.data_ptr())
                self.assertEqual(len(compilations), 1)
            finally:
                torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
