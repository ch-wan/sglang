import pytest
import torch

from sglang.srt.models.looped_transformer import (
    LoopConfig,
    LoopModelAdapter,
    LoopState,
    TrainingFreeLoopPlugin,
    blend_states,
    run_block_anchored,
)


def test_loop_config_accepts_json_and_aliases():
    config = LoopConfig.from_json('{"loop_start": 2, "loop_end": 4, "loop_k": 3}')
    assert config.start == 2
    assert config.end == 4
    assert config.k == 3


def test_loop_plugin_runs_model_independent_adapter():
    batch = type("Batch", (), {})()
    plugin = TrainingFreeLoopPlugin()
    state = LoopState(torch.tensor([[1.0]]), torch.zeros(1, 1))
    adapter = LoopModelAdapter(
        num_layers=4,
        start_layer=0,
        end_layer=4,
        is_first_rank=True,
        is_last_rank=True,
        is_prefill=True,
        is_mixed=False,
        initial_state=lambda: state,
        run_layer=lambda _layer_id, current: LoopState(
            current.stream() + 1.0, torch.zeros_like(current.stream())
        ),
        finalize=lambda current: current.stream(),
    )
    result = plugin.run(LoopConfig(start=1, end=3, k=2), adapter, batch)
    torch.testing.assert_close(result, torch.tensor([[3.0]]))


def test_loop_plugin_allows_decode_full_trial_cache_writes():
    batch = type("Batch", (), {})()
    plugin = TrainingFreeLoopPlugin()
    adapter = LoopModelAdapter(
        num_layers=2,
        start_layer=0,
        end_layer=2,
        is_first_rank=True,
        is_last_rank=True,
        is_prefill=False,
        is_mixed=False,
        initial_state=lambda: LoopState(torch.ones(1, 1), torch.zeros(1, 1)),
        run_layer=lambda _layer_id, current: LoopState(
            current.stream() + 1.0, torch.zeros_like(current.stream())
        ),
        finalize=lambda current: current.stream(),
    )
    result = plugin.run(
        LoopConfig(start=0, end=1, k=2, decode_mode="full"), adapter, batch
    )
    torch.testing.assert_close(result, torch.tensor([[2.0]]))
    assert getattr(batch, "_loop_decode_full", False) is False


def test_blend_states_preserves_residual_stream():
    base = LoopState(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
    target = LoopState(torch.tensor([[5.0, 6.0]]), torch.tensor([[7.0, 8.0]]))

    blended = blend_states(base, target, 0.25)

    torch.testing.assert_close(blended.stream(), torch.tensor([[6.0, 8.0]]))
    torch.testing.assert_close(blended.residual, target.residual)


def test_block_anchored_k1_is_natural_pass():
    state = LoopState(torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    calls = []

    def run_block(current):
        calls.append(getattr(batch, "_loop_disable_kv_cache", False))
        return LoopState(current.stream() + 1.0, torch.zeros_like(current.stream()))

    batch = type("Batch", (), {})()
    result = run_block_anchored(state, run_block, 1, 0.5, batch)

    torch.testing.assert_close(result.stream(), torch.tensor([[4.0]]))
    assert calls == [False]


def test_block_anchored_disables_cache_for_trial_passes():
    state = LoopState(torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    batch = type("Batch", (), {})()
    calls = []

    def run_block(current):
        calls.append(getattr(batch, "_loop_disable_kv_cache", False))
        return LoopState(current.stream() + 1.0, torch.zeros_like(current.stream()))

    run_block_anchored(state, run_block, 3, 0.5, batch)

    assert calls == [False, True, True]
    assert getattr(batch, "_loop_disable_kv_cache", False) is False


@pytest.mark.parametrize("beta", [-0.1, 1.1])
def test_block_anchored_rejects_invalid_anchor(beta):
    state = LoopState(torch.ones(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="anchor beta"):
        run_block_anchored(state, lambda current: current, 2, beta, object())
