"""Complete dense handoffs and compatibility consumers of ordinary tensors."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator_binding import (
    AttentionBoundary,
    DenseLayerBoundaryRules,
    FFNExecution,
    FFNReduction,
)
from sglang.srt.layers.communicator_layout import ResidualState
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
RULES = DenseLayerBoundaryRules(False, True)


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)


class Norm(torch.nn.Module):
    def forward(self, x, residual=None, addition=None):
        if residual is None:
            return rms(x)
        residual.add_(x)
        if addition is not None:
            residual.add_(addition)
        return rms(residual), residual

    def forward_with_allreduce_fusion(self, x, residual, use_attn_tp_group):
        assert not use_attn_tp_group
        return self(x * 2, residual)


@contextmanager
def environment(dp=1, tp=1, rank=0):
    d, t = divmod(rank, tp)
    with (
        get_context().override_server_args(tp_size=dp * tp, dp_size=dp),
        get_parallel().override(
            world_rank=rank,
            tp_rank=rank,
            tp_size=dp * tp,
            attn_dp_rank=d,
            attn_dp_size=dp,
            attn_tp_rank=t,
            attn_tp_size=tp,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_group=SimpleNamespace(unique_name="tp", ranks=list(range(dp * tp))),
            attn_tp_group=SimpleNamespace(
                unique_name=f"atp:{d}", ranks=list(range(d * tp, (d + 1) * tp))
            ),
        ),
        patch.object(comm, "get_moe_cp_size", return_value=1),
        patch.object(comm, "apply_aiter_all_reduce_fusion", return_value=False),
        patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=False),
    ):
        yield


def build(rules=RULES):
    mode = comm.ScatterMode
    return comm.LayerCommunicator(
        comm.LayerScatterModes(
            mode.TP_ATTN_FULL,
            mode.TP_ATTN_FULL,
            mode.FULL,
            mode.TP_ATTN_FULL,
            mode.TP_ATTN_FULL,
        ),
        Norm(),
        Norm(),
        boundary_rules=rules,
    )


def batch(counts):
    return SimpleNamespace(
        global_num_tokens_cpu=counts,
        dp_padding_mode=DpPaddingMode.SUM_LEN,
        forward_mode=ForwardMode.EXTEND,
    )


class TestPrepareAttention(unittest.TestCase):
    def test_execution_and_residual_descriptions(self):
        for dp, execution, expected in (
            (1, FFNExecution(), AttentionBoundary.LOCAL),
            (2, FFNExecution(), AttentionBoundary.SCATTER),
            (1, FFNExecution(FFNReduction.NEXT_PREPARE), AttentionBoundary.REDUCE),
            (
                2,
                FFNExecution(FFNReduction.POSTPROCESS),
                AttentionBoundary.REDUCE_SCATTER,
            ),
        ):
            with self.subTest(dp=dp, execution=execution), environment(dp, 2):
                fwd = batch([4] * dp)
                fwd.dp_padding_mode = DpPaddingMode.MAX_LEN
                binding = RULES.bind(fwd, valid_tokens=[3] * dp, execution=execution)
                self.assertIs(binding.attention_boundary, expected)
                self.assertIs(
                    binding.attention_input(ResidualState.SEPARATE),
                    binding.prepare_attn_source,
                )
                for state in (ResidualState.ABSENT, ResidualState.MERGED):
                    if expected is AttentionBoundary.REDUCE:
                        with self.assertRaises(ValueError):
                            binding.attention_input(state)
                    else:
                        source = binding.attention_input(state)
                        self.assertIs(source.residual_state, state)
                        self.assertIsNone(source.residual)
                        self.assertEqual(
                            source.hidden, binding.prepare_attn_target.hidden
                        )
                if execution == FFNExecution():
                    self.assertIs(
                        build()._attention_boundary, binding.attention_boundary
                    )

    def test_local_first_merged_separate_empty_and_extra_residual(self):
        with environment():
            layer = build()
            for rows, separate in ((3, False), (2, True), (0, True), (1, False)):
                x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) + 1
                residual = torch.full_like(x, 2) if separate else None
                extra = torch.full_like(x, 0.5)
                expected = x + 2.5 if separate else x
                out, out_r = layer.prepare_attn(
                    x, residual, batch([rows]), post_residual_addition=extra
                )
                torch.testing.assert_close(out, rms(expected))
                torch.testing.assert_close(out_r, expected)
                self.assertIs(out_r, residual if separate and rows else x)

    def test_dp_postprocess_then_prepare_does_not_reduce_or_add_twice(self):
        for tp, counts in ((1, [3, 1]), (2, [4, 2]), (2, [0, 4]), (2, [0, 0])):
            full = (
                torch.arange(sum(counts) * 4, dtype=torch.float32).reshape(-1, 4) / 10
            )
            for rank in range(2 * tp):
                d = rank // tp
                local = full[sum(counts[:d]) : sum(counts[: d + 1])]
                with (
                    self.subTest(tp=tp, counts=counts, rank=rank),
                    environment(2, tp, rank),
                ):
                    layer = build()
                    residual = torch.ones_like(local)
                    layer._communicate_summable_tensor_pair_fn = Mock(
                        side_effect=AssertionError("legacy exit")
                    )
                    layer._communicate_simple_fn = Mock(
                        side_effect=AssertionError("legacy entry")
                    )
                    with (
                        patch.object(
                            comm,
                            "get_local_dp_buffer",
                            return_value=torch.empty_like(local),
                        ),
                        patch.object(comm, "get_tp_group", return_value=None),
                        patch.object(
                            comm,
                            "dp_scatter",
                            side_effect=lambda out, src, fwd: out.copy_(local),
                        ) as scatter,
                        patch.object(
                            comm, "should_use_dp_reduce_scatterv", return_value=True
                        ) as policy,
                        patch.object(
                            comm, "moe_tensor_model_parallel_all_reduce"
                        ) as ar,
                        get_forward().scoped(
                            fuse_mlp_allreduce=True, mlp_reduce_scatter=True
                        ),
                    ):
                        h, r = layer.postprocess_layer(full, residual, batch(counts))
                        torch.testing.assert_close(r, torch.ones_like(local))
                        out, out_r = layer.prepare_attn(h, r, batch(counts))
                        torch.testing.assert_close(out, rms(local + 1))
                        torch.testing.assert_close(out_r, local + 1)
                        scatter.assert_called_once()
                        policy.assert_not_called()
                        ar.assert_not_called()

    def test_legacy_deferred_reduction_fused_and_plain(self):
        for fused in (False, True):
            with self.subTest(fused=fused), environment(tp=2):
                layer = build(rules=None)
                x, residual = torch.ones(2, 4), torch.full((2, 4), 3.0)
                x._sglang_needs_allreduce_fusion = True
                with (
                    patch.object(
                        comm, "apply_flashinfer_allreduce_fusion", return_value=fused
                    ),
                    patch.object(
                        comm,
                        "moe_tensor_model_parallel_all_reduce",
                        side_effect=lambda h: h * 2,
                    ) as ar,
                    get_forward().scoped(fuse_mlp_allreduce=False),
                ):
                    out, out_r = layer.prepare_attn(x, residual, batch([2]))
                torch.testing.assert_close(out_r, torch.full_like(x, 5))
                torch.testing.assert_close(out, rms(out_r))
                self.assertEqual(ar.call_count, int(not fused))

    def test_quant_tuple_callback_and_aux_snapshot(self):
        with environment():
            layer = build()
            layer.qkv_latent_func = Mock()
            layer.input_layernorm.weight = torch.nn.Parameter(torch.ones(4))
            layer.input_layernorm.variance_epsilon = 1e-5
            q, scale = torch.ones(2, 4), torch.ones(2, 1)
            residual = torch.ones(2, 4)
            extra = torch.full_like(residual, 2)
            expected_residual = residual + extra + 1

            def quant(x, weight, eps, residual=None):
                if residual is None:
                    return q, scale
                return (q, scale), residual + x

            with (
                patch.object(comm, "_use_aiter", True),
                patch.object(
                    comm, "_fused_rmsnorm_fp8_per_token_quant", side_effect=quant
                ),
                get_forward().scoped(attn_inputs=None),
            ):
                captured = []
                out, out_r = layer.prepare_attn_and_capture_last_layer_outputs(
                    torch.ones(2, 4),
                    residual,
                    batch([2]),
                    captured,
                    post_residual_addition=extra,
                    quant_format="fp8_per_token",
                )
                self.assertIs(out[0], q)
                self.assertIs(out[1], scale)
                self.assertIs(get_forward().attn_inputs.hidden_states_local, out)
                layer.qkv_latent_func.assert_not_called()  # registration remains lazy
                torch.testing.assert_close(out_r, expected_residual)
                out_r.add_(10)
                torch.testing.assert_close(captured[0], expected_residual)
                out, out_r = layer.prepare_attn(
                    torch.ones(2, 4), None, batch([2]), quant_format="fp8_per_token"
                )
                self.assertIs(out[0], q)
                torch.testing.assert_close(out_r, torch.ones(2, 4))

    def test_special_entries_preserve_adapter(self):
        with environment():
            layer = build()
            x = torch.ones(2, 4)
            layer._tp_reduce_scatter = Mock(return_value=(x, None))
            layer._communicate_simple_fn = Mock(
                side_effect=lambda **kw: kw["hidden_states"]
            )
            layer._communicate_summable_tensor_pair_fn = Mock(return_value=(x, None))
            with get_forward().scoped(attn_input_scattered=True):
                layer.prepare_attn(x, None, batch([2]))
                layer.postprocess_layer(x, None, batch([2]))
            layer._tp_reduce_scatter.assert_called_once()
            layer._communicate_simple_fn.assert_called_once()
            layer._communicate_summable_tensor_pair_fn.assert_called_once()
            layer._sp_variant = SimpleNamespace(
                prepare_attn=Mock(return_value=(x, None))
            )
            with get_forward().scoped(sp_active=True):
                layer.prepare_attn(x, x.clone(), batch([2]))
            layer._sp_variant.prepare_attn.assert_called_once()

    def test_qwen2_moe_pp_sender_receiver_and_terminal_norm(self):
        from sglang.srt.models import qwen2_moe as model

        x, residual = torch.ones(2, 4), torch.full((2, 4), 3.0)
        x._sglang_needs_allreduce_fusion = True
        shell = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
            start_layer=0,
            end_layer=0,
            layers=[],
            norm=Norm(),
        )
        fwd = SimpleNamespace(can_run_tbo=False)
        proxy = PPProxyTensors({"hidden_states": x, "residual": residual})
        with (
            environment(tp=2),
            get_parallel().override(moe_ep_size=1, moe_tp_size=2),
            patch.object(
                model,
                "moe_tensor_model_parallel_all_reduce",
                side_effect=lambda h: h * 2,
            ) as reduce,
            patch.object(model, "moe_expert_parallel_all_reduce") as ep,
        ):
            sent = model.Qwen2MoeModel.forward(
                shell, None, None, fwd, pp_proxy_tensors=proxy
            )
            reduce.assert_called_once()
            ep.assert_not_called()
            self.assertFalse(sent["hidden_states"]._sglang_needs_allreduce_fusion)
            layer = build(rules=None)
            with patch.object(comm, "moe_tensor_model_parallel_all_reduce") as again:
                out, out_r = layer.prepare_attn(
                    sent["hidden_states"], sent["residual"].clone(), batch([2])
                )
                again.assert_not_called()
            torch.testing.assert_close(out_r, torch.full_like(x, 5))
            shell.pp_group.is_last_rank = True
            terminal = model.Qwen2MoeModel.forward(
                shell, None, None, fwd, pp_proxy_tensors=sent
            )
            torch.testing.assert_close(terminal, out)

    def test_complete_handoff_compiles_fullgraph(self):
        with environment():
            layer = build()
            graphs = []

            def backend(gm, args):
                graphs.append(gm)
                return gm.forward

            def run(x, r):
                h, r = layer.postprocess_layer(x, r, None)
                return layer.prepare_attn(h, r, None)

            compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
            for n in (2, 5, 3):
                x, r = torch.rand(n, 4), torch.rand(n, 4)
                torch.testing.assert_close(compiled(x, r.clone()), (rms(x + r), x + r))
            self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
