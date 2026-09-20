# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Small runtime helpers for training-free looped transformer inference."""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch


@dataclass(frozen=True)
class LoopConfig:
    """JSON-configurable settings for a training-free transformer loop."""

    start: int
    end: int
    k: int = 1
    mode: str = "block"
    strategy: str = "block_anchored"
    anchor_beta: float = 0.5
    decode_mode: str = "bypass"
    cache_strategy: str = "first"

    @classmethod
    def from_json(cls, raw: str | dict) -> Optional[LoopConfig]:
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"loop-config must be valid JSON: {exc}") from exc
        else:
            value = raw
        if value in (None, {}):
            return None
        if not isinstance(value, dict):
            raise ValueError("loop-config must be a JSON object")
        aliases = {"loop_start": "start", "loop_end": "end", "loop_k": "k"}
        normalized = {aliases.get(key, key): item for key, item in value.items()}
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "loop-config contains unknown fields: " + ", ".join(sorted(unknown))
            )
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"loop window must satisfy 0 <= start < end, got "
                f"[{self.start}, {self.end})"
            )
        if self.k < 1:
            raise ValueError(f"loop k must be >= 1, got {self.k}")
        if self.mode != "block":
            raise ValueError("the first loop implementation only supports mode=block")
        if self.strategy != "block_anchored":
            raise ValueError(
                "the first loop implementation only supports strategy=block_anchored"
            )
        if not 0.0 <= self.anchor_beta <= 1.0:
            raise ValueError(
                f"loop anchor_beta must be in [0, 1], got {self.anchor_beta}"
            )
        if self.decode_mode not in {"bypass", "full"}:
            raise ValueError(
                "the first loop implementation only supports decode_mode=bypass or full"
            )
        if self.cache_strategy != "first":
            raise ValueError(
                "the first loop implementation only supports cache_strategy=first"
            )


@dataclass
class LoopModelAdapter:
    """Model-specific hooks consumed by :class:`TrainingFreeLoopPlugin`.

    A model integration only needs to expose its layer traversal and residual
    state representation. The loop algorithm, validation, and execution
    policy remain model-independent.
    """

    num_layers: int
    start_layer: int
    end_layer: int
    is_first_rank: bool
    is_last_rank: bool
    is_prefill: bool
    is_mixed: bool
    initial_state: Callable[[], LoopState]
    run_layer: Callable[[int, LoopState], LoopState]
    finalize: Callable[[LoopState], object]


class TrainingFreeLoopPlugin:
    """Reusable block-loop plugin for decoder-only model integrations."""

    def run(
        self,
        config: Optional[LoopConfig],
        adapter: LoopModelAdapter,
        forward_batch,
    ):
        if config is None or config.k <= 1:
            return None
        if adapter.is_mixed or (
            not adapter.is_prefill and config.decode_mode != "full"
        ):
            return None
        if not (0 <= config.start < config.end <= adapter.num_layers):
            raise ValueError(
                f"invalid loop window [{config.start}, {config.end}) for "
                f"{adapter.num_layers} layers"
            )
        if config.start < adapter.start_layer or config.end > adapter.end_layer:
            raise ValueError(
                "the loop window must be contained in one pipeline-parallel rank: "
                f"window=[{config.start}, {config.end}), "
                f"local=[{adapter.start_layer}, {adapter.end_layer})"
            )

        initial = adapter.initial_state()

        def run_block(state: LoopState) -> LoopState:
            for layer_id in range(config.start, config.end):
                state = adapter.run_layer(layer_id, state)
            return state

        decode_full = not adapter.is_prefill and config.decode_mode == "full"
        old_decode_full = getattr(forward_batch, "_loop_decode_full", False)
        forward_batch._loop_decode_full = decode_full
        try:
            state = run_block_anchored(
                initial,
                run_block,
                K=config.k,
                anchor_beta=config.anchor_beta,
                forward_batch=forward_batch,
                trial_cache_writes=decode_full,
            )
        finally:
            forward_batch._loop_decode_full = old_decode_full
        return adapter.finalize(state)


@dataclass
class LoopState:
    """The split hidden/residual state used by SGLang decoder layers."""

    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]

    def stream(self) -> torch.Tensor:
        if self.residual is None:
            return self.hidden_states
        return self.hidden_states + self.residual


def clone_state(state: LoopState) -> LoopState:
    """Detach a state before passing it through a potentially in-place layer."""

    return LoopState(
        hidden_states=state.hidden_states.clone(),
        residual=state.residual.clone() if state.residual is not None else None,
    )


def blend_states(base: LoopState, target: LoopState, alpha: float) -> LoopState:
    """Blend complete residual streams while preserving target layout.

    The next decoder layer consumes the sum of hidden_states and residual.
    Keeping the target residual tensor and adjusting hidden_states preserves
    that sum without changing LayerCommunicator's expected tensor layout.
    """

    stream = torch.lerp(base.stream(), target.stream(), alpha)
    if target.residual is None:
        return LoopState(hidden_states=stream, residual=None)
    return LoopState(
        hidden_states=stream - target.residual,
        residual=target.residual.clone(),
    )


@contextmanager
def disable_kv_cache(forward_batch) -> Iterator[None]:
    """Temporarily suppress KV writes made by a loop trial pass."""

    old_value = getattr(forward_batch, "_loop_disable_kv_cache", False)
    forward_batch._loop_disable_kv_cache = True
    try:
        yield
    finally:
        forward_batch._loop_disable_kv_cache = old_value


def run_block_anchored(
    initial: LoopState,
    run_block: Callable[[LoopState], LoopState],
    K: int,
    anchor_beta: float,
    forward_batch,
    trial_cache_writes: bool = False,
) -> LoopState:
    """Run the paper's block-anchored damped Euler method.

    The natural pass writes the canonical KV entry. Extra trial passes are
    evaluated with KV writes disabled.
    """

    if K < 1:
        raise ValueError(f"loop K must be >= 1, got {K}")
    if not 0.0 <= anchor_beta <= 1.0:
        raise ValueError(f"loop anchor beta must be in [0, 1], got {anchor_beta}")

    natural = run_block(clone_state(initial))
    if K == 1:
        return natural

    h = blend_states(initial, natural, 1.0 / K)
    trial_context = (
        nullcontext() if trial_cache_writes else disable_kv_cache(forward_batch)
    )
    with trial_context:
        for _ in range(1, K):
            proposal = run_block(clone_state(h))
            h = blend_states(h, proposal, 1.0 / K)

    return blend_states(h, natural, anchor_beta)
