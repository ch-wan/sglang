"""MoE exits contain token ownership and TP sums, never pending EP work."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator_binding import FFNReduction
from sglang.srt.layers.communicator_layout import Partial, validate_layouts
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe import communicator as moe_comm
from sglang.srt.layers.moe.communicator import MoEBoundaryPlan, MoEOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.models import qwen3_moe as model
from sglang.srt.models.mellum import MellumDecoderLayer
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@contextmanager
def topology(ep=2, mtp=1, dp=1, rank=0, backend="none"):
    world = ep * mtp
    atp = world // dp
    d, t = divmod(rank, atp)
    with (
        get_context().override_server_args(
            tp_size=world, dp_size=dp, moe_a2a_backend=backend
        ),
        get_parallel().override(
            world_rank=rank,
            tp_rank=rank,
            tp_size=world,
            attn_dp_rank=d,
            attn_dp_size=dp,
            attn_tp_rank=t,
            attn_tp_size=atp,
            attn_cp_rank=0,
            attn_cp_size=1,
            moe_ep_size=ep,
            moe_tp_size=mtp,
            moe_dp_size=1,
            tp_group=SimpleNamespace(unique_name="tp", ranks=list(range(world))),
            attn_tp_group=SimpleNamespace(
                unique_name=f"attn:{d}", ranks=list(range(d * atp, (d + 1) * atp))
            ),
        ),
        patch.object(model, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)),
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend(backend),
        ),
        patch.object(
            model, "is_deepep_class_backend", return_value=backend in ("deepep", "pplx")
        ),
        patch.object(
            model, "should_use_flashinfer_cutlass_moe_fp4_allgather", return_value=False
        ),
        patch.object(model, "should_use_dp_reduce_scatterv", return_value=False),
        patch(
            "sglang.srt.layers.moe.utils.should_use_dp_reduce_scatterv",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.utils.should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ),
    ):
        yield


def batch(counts, max_len=True):
    return SimpleNamespace(
        global_num_tokens_cpu=counts,
        dp_padding_mode=DpPaddingMode.MAX_LEN if max_len else DpPaddingMode.SUM_LEN,
    )


def plan(ep=2, scattered=False, fuse=False, rs=False, rsv=False):
    return MoEBoundaryPlan.select(
        ep_size=ep,
        scattered=scattered,
        fuse_next=fuse,
        reduce_scatter=rs,
        reduce_scatterv=rsv,
    )


def block(ep=2, tp=1):
    obj = model.Qwen3MoeSparseMoeBlock.__new__(model.Qwen3MoeSparseMoeBlock)
    torch.nn.Module.__init__(obj)
    obj.ep_size, obj.tp_size = ep, tp
    obj.gate = Mock(side_effect=lambda x: (x, None))
    obj.topk = Mock(return_value=object())
    obj.experts = Mock(side_effect=lambda x, topk: x.clone())
    return obj


class TestQwen3MoeBoundary(unittest.TestCase):
    def test_policy_and_producer_scope_restore(self):
        with topology(), get_forward().scoped(fuse_mlp_allreduce=True):
            for ep, fuse, rs, rsv, scattered, expected, reduction in (
                (2, True, False, False, False, MoEOutput.GLOBAL, FFNReduction.COMPUTE),
                (
                    2,
                    False,
                    True,
                    False,
                    False,
                    MoEOutput.ATTENTION_LOCAL,
                    FFNReduction.COMPUTE,
                ),
                (
                    2,
                    False,
                    False,
                    True,
                    False,
                    MoEOutput.ATTENTION_LOCAL,
                    FFNReduction.COMPUTE,
                ),
                (
                    2,
                    False,
                    False,
                    False,
                    True,
                    MoEOutput.SCATTERED,
                    FFNReduction.COMPUTE,
                ),
                (
                    1,
                    True,
                    False,
                    False,
                    False,
                    MoEOutput.GLOBAL,
                    FFNReduction.NEXT_PREPARE,
                ),
                (
                    1,
                    False,
                    True,
                    False,
                    False,
                    MoEOutput.GLOBAL,
                    FFNReduction.POSTPROCESS,
                ),
            ):
                p = plan(ep, scattered, fuse, rs, rsv)
                self.assertIs(p.output, expected)
                self.assertIs(p.ffn.reduction, reduction)
                with self.assertRaisesRegex(RuntimeError, "scope exit"):
                    with p.scope():
                        self.assertEqual(
                            get_forward().fuse_mlp_allreduce,
                            reduction is FFNReduction.NEXT_PREPARE,
                        )
                        raise RuntimeError("scope exit")
                self.assertTrue(get_forward().fuse_mlp_allreduce)
            with self.assertRaises(ValueError):
                plan(ep=1, fuse=True, rs=True)

    def test_ep_then_tp_completed_inside_moe_and_no_residual_added(self):
        # E0/T0=1, E0/T1=2, E1/T0=3, E1/T1=4 -> EP=4/6 -> TP=10.
        for rows in (3, 0):
            with topology(ep=2, mtp=2):
                obj = block(2, 2)
                events = []
                p = plan(ep=2, fuse=True)
                with (
                    patch.object(
                        model,
                        "moe_expert_parallel_all_reduce",
                        side_effect=lambda x: (events.append("ep"), x * 4)[1],
                    ),
                    patch.object(
                        model,
                        "moe_tensor_model_parallel_all_reduce",
                        side_effect=lambda x: (events.append("tp"), x * 2.5)[1],
                    ),
                    p.scope(),
                ):
                    out = obj(torch.ones(rows, 4), boundary_plan=p)
                self.assertEqual(events, ["ep", "tp"])
                torch.testing.assert_close(out, torch.full((rows, 4), 10.0))
                self.assertFalse(hasattr(out, "_sglang_needs_allreduce_fusion"))
                if not rows:
                    obj.gate.assert_not_called()
                    obj.topk.empty_topk_output.assert_called_once()

    def test_dp_expert_combine_precedes_return_and_only_runs_once(self):
        for rsv in (False, True):
            with self.subTest(rsv=rsv), topology(dp=2):
                obj = block()
                p = plan(rs=not rsv, rsv=rsv)
                x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
                expected = x[:2] * 3
                group = SimpleNamespace(
                    reduce_scatterv=Mock(
                        side_effect=lambda src, output, sizes: output.copy_(expected)
                    )
                )
                with (
                    patch.object(model, "moe_expert_parallel_all_reduce") as ep,
                    patch.object(moe_comm, "get_tp_group", return_value=group),
                    patch.object(
                        moe_comm, "get_dp_global_num_tokens", return_value=[2, 2]
                    ),
                    patch.object(
                        moe_comm, "get_local_dp_buffer", return_value=torch.empty(2, 4)
                    ),
                    patch.object(
                        moe_comm,
                        "dp_reduce_scatter_tensor",
                        side_effect=lambda out, src: out.copy_(expected),
                    ) as rs,
                    p.scope(),
                ):
                    out = obj(x, boundary_plan=p)
                    ep.assert_not_called()
                    self.assertEqual(rs.call_count, int(not rsv))
                    self.assertEqual(group.reduce_scatterv.call_count, int(rsv))
                torch.testing.assert_close(out, expected)
                self.assertIs(p.output, MoEOutput.ATTENTION_LOCAL)

    def test_a2a_combine_is_not_reduced_again(self):
        for backend in ("none", "deepep", "flashinfer", "pplx", "flashinfer_megamoe"):
            with self.subTest(backend=backend), topology(backend=backend):
                if backend == "none":
                    continue
                obj = block()
                complete = torch.arange(8, dtype=torch.float32).reshape(2, 4)
                obj.forward_deepep = Mock(return_value=complete)
                obj.experts = Mock(return_value=complete)
                p = plan(scattered=True)
                with (
                    patch.object(model, "moe_expert_parallel_all_reduce") as ep,
                    patch.object(model, "moe_tensor_model_parallel_all_reduce") as tp,
                    p.scope(),
                ):
                    out = obj(torch.ones(2, 4), batch([4]), boundary_plan=p)
                self.assertIs(
                    out, complete
                ) if backend == "deepep" else torch.testing.assert_close(out, complete)
                ep.assert_not_called()
                tp.assert_not_called()

    def test_bind_complete_and_scattered_ragged_outputs(self):
        for scattered, rs, rsv, physical, valid in (
            (False, False, False, [4, 2], [3, 1]),
            (False, True, False, [4, 4], [3, 1]),
            (False, False, True, [4, 2], [3, 1]),
            (True, False, False, [4, 2], [3, 1]),
        ):
            layouts = []
            ep = 2 if rsv else 4
            for rank in range(ep):
                with topology(ep=ep, dp=2, rank=rank):
                    p = plan(ep=ep, scattered=scattered, rs=rs, rsv=rsv)
                    b = p.bind_output(batch(physical, max_len=rs), valid_tokens=valid)
                    self.assertFalse(
                        any(
                            isinstance(g.placement, Partial)
                            for g in b.hidden.placements
                        )
                    )
                    layouts.append(b)
            validate_layouts([b.hidden for b in layouts], num_tokens=sum(valid))
            validate_layouts([b.residual for b in layouts], num_tokens=sum(valid))
        with topology(ep=1, mtp=2):
            b = plan(ep=1, fuse=True).bind_output(batch([4]), valid_tokens=[3])
            self.assertTrue(
                any(isinstance(g.placement, Partial) for g in b.hidden.placements)
            )
        with topology(ep=2), get_parallel().override(moe_dp_size=2):
            with self.assertRaises(NotImplementedError):
                plan(ep=2).bind_output(batch([4]), valid_tokens=[3])
        with topology(ep=2), get_context().override_server_args(dwdp_size=2):
            with self.assertRaises(NotImplementedError):
                plan(ep=2).bind_output(batch([4]), valid_tokens=[3])
        # Pending EP work must not be relabeled as a pure TP partial.
        with topology(ep=2):
            with self.assertRaises(ValueError):
                plan(ep=1, fuse=True).bind_output(batch([4]), valid_tokens=[3])

    def test_mellum_and_special_path_selection(self):
        shell = SimpleNamespace(
            is_layer_sparse=True,
            layer_scatter_modes=SimpleNamespace(mlp_mode=comm.ScatterMode.FULL),
        )
        for cls in (model.Qwen3MoeDecoderLayer, MellumDecoderLayer):
            with topology():
                p = cls.plan_moe_boundary(shell, fuse_next=True, reduce_scatter=False)
                self.assertIs(p.ffn.reduction, FFNReduction.COMPUTE)
                shell.is_layer_sparse = False
                self.assertIsNone(
                    cls.plan_moe_boundary(shell, fuse_next=True, reduce_scatter=False)
                )
                shell.is_layer_sparse = True
                for overrides in (
                    {"attn_cp_size": 2},
                    {"moe_dp_size": 2},
                ):
                    with get_parallel().override(**overrides):
                        self.assertIsNone(
                            cls.plan_moe_boundary(
                                shell, fuse_next=False, reduce_scatter=False
                            )
                        )
                for flags in ({"sp_active": True}, {"attn_input_scattered": True}):
                    with get_forward().scoped(**flags):
                        self.assertIsNone(
                            cls.plan_moe_boundary(
                                shell, fuse_next=False, reduce_scatter=False
                            )
                        )
        shell.layer_scatter_modes.mlp_mode = comm.ScatterMode.SCATTERED
        for backend in ("deepep", "flashinfer", "pplx", "customized"):
            with topology(backend=backend):
                p = model.Qwen3MoeDecoderLayer.plan_moe_boundary(
                    shell, fuse_next=False, reduce_scatter=False
                )
                if backend == "customized":
                    self.assertIsNone(p)
                else:
                    self.assertIs(p.output, MoEOutput.SCATTERED)

    def test_real_decoder_consumes_plan_after_flag_scope(self):
        for cls in (model.Qwen3MoeDecoderLayer, MellumDecoderLayer):
            for rs in (False, True):
                with self.subTest(cls=cls, rs=rs), topology(dp=2):
                    x, residual = torch.ones(4, 4), torch.ones(2, 4)
                    communicator = SimpleNamespace(
                        prepare_attn_and_capture_last_layer_outputs=Mock(
                            return_value=(x, residual)
                        ),
                        prepare_mlp=Mock(return_value=(x, residual)),
                        should_fuse_mlp_allreduce_with_next_layer=Mock(
                            return_value=False
                        ),
                        should_use_reduce_scatter=Mock(return_value=rs),
                        postprocess_completed_ffn=Mock(return_value=(x[:2], residual)),
                        postprocess_layer=Mock(
                            side_effect=AssertionError("partial-value postprocess")
                        ),
                    )
                    shell = SimpleNamespace(
                        layer_communicator=communicator,
                        self_attn=Mock(return_value=x),
                        plan_moe_boundary=lambda **kw: plan(rs=rs),
                        mlp=Mock(return_value=x[:2] if rs else x),
                    )
                    h, r = cls.forward(shell, None, x, batch([2, 2]), residual)
                    torch.testing.assert_close(h, x[:2])
                    self.assertIs(r, residual)
                    self.assertEqual(
                        communicator.postprocess_completed_ffn.call_count, int(not rs)
                    )
                    self.assertFalse(get_forward().mlp_reduce_scatter)
                    self.assertIsInstance(
                        shell.mlp.call_args.kwargs["boundary_plan"], MoEBoundaryPlan
                    )

    def test_plan_and_wrapper_compile_fullgraph(self):
        class Gate(torch.nn.Module):
            def forward(self, x):
                return x, None

        class Route(torch.nn.Module):
            def forward(self, x, logits):
                return logits

        class Experts(torch.nn.Module):
            def forward(self, x, topk):
                return x.sin()

        with (
            topology(ep=1),
            patch.object(model, "is_deepep_class_backend", lambda: False),
            patch.object(model, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE),
        ):
            obj = block(1, 1)
            obj.gate, obj.topk, obj.experts = Gate(), Route(), Experts()
            graphs = []

            def backend(gm, args):
                graphs.append(gm)
                return gm.forward

            def run(x):
                p = MoEBoundaryPlan.select(
                    ep_size=1,
                    scattered=False,
                    fuse_next=False,
                    reduce_scatter=False,
                    reduce_scatterv=False,
                )
                with p.scope():
                    return obj(x, boundary_plan=p)

            compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
            for n in (2, 5, 3):
                x = torch.randn(n, 4)
                torch.testing.assert_close(compiled(x), x.sin())
            self.assertEqual(len(graphs), 1)

    def test_pure_tp_remains_partial(self):
        with topology(ep=1, mtp=2):
            obj = block(1, 2)
            p = plan(ep=1, fuse=True)
            with (
                patch.object(model, "moe_tensor_model_parallel_all_reduce") as reduce,
                p.scope(),
            ):
                x = torch.ones(2, 4)
                out = obj(x, boundary_plan=p)
                reduce.assert_not_called()
                torch.testing.assert_close(out, x)


if __name__ == "__main__":
    unittest.main()
