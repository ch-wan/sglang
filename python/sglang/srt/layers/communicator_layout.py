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
"""Host-side descriptions of attention/FFN preparation boundaries.

These types describe values, not collective schedules or DTensors. Bind them
from the topology and token domain of one forward/microbatch. A tensor does not
carry or infer its own layout. No existing communicator path uses these yet.

``validate_layouts`` is an explicit, exhaustive semantic check for tests/debugging:
it needs all participating ranks and expands token spans. It must not run in a
forward or graph capture. Production binding/storage belongs to the preparation
adapter; there is no metadata collective or process-group handle cache here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class ParallelAxis(Enum):
    ATTN_DP = auto()
    ATTN_TP = auto()
    ATTN_CP = auto()
    FFN_DP = auto()
    FFN_TP = auto()


@dataclass(frozen=True)
class ParallelGroup:
    """An ordered rank group, including the coordinator's identity/generation.

    Logical ownership groups (e.g. attention DP) may be declared explicitly;
    their presence does not imply that a collective exists for that group.
    Equal sizes, member sets or roles alone do not identify the same group.
    """

    name: str
    ranks: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if not self.name or not self.ranks:
            raise ValueError("A group needs a name and at least one rank")
        if min(self.ranks) < 0 or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("Group ranks must be distinct nonnegative global ranks")

    @classmethod
    def from_coordinator(cls, group: GroupCoordinator) -> ParallelGroup:
        """Read the current coordinator; retain no handle to it."""
        return cls(group.unique_name, tuple(group.ranks))


@dataclass(frozen=True)
class Shard:
    dim: int = 0

    def __post_init__(self):
        if self.dim != 0:
            raise ValueError(
                "Preparation layouts currently support token Shard(0) only"
            )


@dataclass(frozen=True)
class Replicate:
    pass


@dataclass(frozen=True)
class Partial:
    reduce_op: str = "sum"

    def __post_init__(self):
        if self.reduce_op != "sum":
            raise ValueError("Preparation layouts currently support Partial(sum) only")


@dataclass(frozen=True)
class GroupPlacement:
    axis: ParallelAxis
    group: ParallelGroup
    placement: Shard | Replicate | Partial


@dataclass(frozen=True)
class TokenSpan:
    """Map consecutive local rows to strided IDs in the boundary's token domain."""

    token_start: int
    length: int
    local_start: int = 0
    stride: int = 1

    def __post_init__(self):
        if min(self.token_start, self.length, self.local_start) < 0 or self.stride < 1:
            raise ValueError(
                "Token spans require nonnegative offsets/length and positive stride"
            )


@dataclass(frozen=True)
class TokenPartition:
    """The joint token mapping, independent of the number/order of Shard bindings.

    Unmapped rows are padding. Multiple spans represent request boundaries,
    zigzag chunks and arbitrary chunk ordering without storing one ID per row.
    An empty shard may still have a nonzero padded size.
    """

    local_size: int
    spans: tuple[TokenSpan, ...]

    def __post_init__(self):
        object.__setattr__(self, "spans", tuple(self.spans))
        if self.local_size < 0:
            raise ValueError("Local padded size must be nonnegative")
        if any(s.local_start + s.length > self.local_size for s in self.spans):
            raise ValueError("Token span exceeds local padded size")
        intervals = sorted(
            (s.local_start, s.local_start + s.length) for s in self.spans if s.length
        )
        end = 0
        for start, stop in intervals:
            if start < end or stop > self.local_size:
                raise ValueError("Token spans overlap or exceed local padded size")
            end = stop

    def token_ids(self) -> tuple[int | None, ...]:
        """Expand for semantic validation only; never inspect a device tensor."""
        ids = [None] * self.local_size
        seen = set()
        for span in self.spans:
            for i in range(span.length):
                token = span.token_start + i * span.stride
                if token in seen:
                    raise ValueError("A rank cannot hold the same logical token twice")
                ids[span.local_start + i] = token
                seen.add(token)
        return tuple(ids)


@dataclass(frozen=True)
class TensorLayout:
    rank: int
    placements: tuple[GroupPlacement, ...]
    tokens: TokenPartition

    def __post_init__(self):
        object.__setattr__(self, "placements", tuple(self.placements))
        if self.rank < 0:
            raise ValueError("Layout rank must be nonnegative")
        axes = set()
        partials = []
        for binding in self.placements:
            if not isinstance(binding.axis, ParallelAxis):
                raise ValueError("Unknown preparation axis")
            if not isinstance(binding.placement, (Shard, Replicate, Partial)):
                raise ValueError("Unknown placement")
            if binding.axis in axes:
                raise ValueError("A layout may bind each role only once")
            axes.add(binding.axis)
            if self.rank not in binding.group.ranks:
                raise ValueError("A rank may only describe groups it belongs to")
            if isinstance(binding.placement, Partial):
                partials.append(binding)
        if len(partials) > 1:
            raise ValueError(
                "Only one partial group is supported; aliases must not reduce twice"
            )
        for i, left in enumerate(self.placements):
            for right in self.placements[i + 1 :]:
                if (
                    set(left.group.ranks) == set(right.group.ranks)
                    and left.placement != right.placement
                ):
                    raise ValueError(
                        "Aliased rank groups have contradictory placements"
                    )
        object.__setattr__(
            self,
            "placements",
            tuple(sorted(self.placements, key=lambda b: b.axis.value)),
        )


class ResidualState(Enum):
    ABSENT = auto()
    SEPARATE = auto()
    MERGED = auto()


@dataclass(frozen=True)
class BoundaryLayout:
    hidden: TensorLayout
    residual_state: ResidualState
    residual: TensorLayout | None = None

    def __post_init__(self):
        if not isinstance(self.residual_state, ResidualState):
            raise ValueError("Residual state must be explicit")
        if (self.residual_state is ResidualState.SEPARATE) != (
            self.residual is not None
        ):
            raise ValueError("Only a separate residual has its own layout")
        if self.residual is not None and self.residual.rank != self.hidden.rank:
            raise ValueError("Hidden and residual must describe the same local rank")


def validate_layouts(layouts: Sequence[TensorLayout], *, num_tokens: int) -> None:
    """Validate a complete rank view over one shared logical token domain.

    Missing roles make no claims; replicas must be explicitly connected by
    Replicate bindings. A Partial binding describes distinct contributions at
    identical token positions, not interchangeable copies. Multiple independent
    partial decompositions of a token are outside this initial contract.
    """
    if num_tokens < 0 or not layouts:
        raise ValueError("Validation needs layouts and a nonnegative token count")
    by_rank = {layout.rank: layout for layout in layouts}
    if len(by_rank) != len(layouts):
        raise ValueError("Provide exactly one layout per participating rank")
    ids = {rank: layout.tokens.token_ids() for rank, layout in by_rank.items()}
    holders: dict[int, set[int]] = {}
    for rank, tokens in ids.items():
        for token in tokens:
            if token is not None:
                if token >= num_tokens:
                    raise ValueError("Token ID exceeds the shared token domain")
                holders.setdefault(token, set()).add(rank)
    if len(holders) != num_tokens:
        raise ValueError("Layouts do not cover the complete token domain")

    replicas = {rank: {rank} for rank in by_rank}
    partials: dict[int, ParallelGroup] = {}
    checked = set()
    for rank, layout in by_rank.items():
        for binding in layout.placements:
            if isinstance(binding.placement, Partial):
                partials[rank] = binding.group
            key = (binding.axis, binding.group)
            if key in checked:
                continue
            checked.add(key)
            members = binding.group.ranks
            for peer in members:
                if peer not in by_rank or binding not in by_rank[peer].placements:
                    raise ValueError(
                        "Every group member must declare the same ordered group and placement"
                    )
            if isinstance(binding.placement, Shard):
                owned = set()
                for peer in members:
                    tokens = {t for t in ids[peer] if t is not None}
                    if owned & tokens:
                        raise ValueError("Shard members cannot own overlapping tokens")
                    owned.update(tokens)
            else:
                if any(ids[peer] != ids[rank] for peer in members):
                    raise ValueError(
                        "Replicate/Partial members must align tokens and padding row by row"
                    )
                if isinstance(binding.placement, Replicate):
                    for peer in members:
                        replicas[peer].update(members)

    for ranks in holders.values():
        reductions = {partials[r] for r in ranks if r in partials}
        if reductions:
            if len(reductions) != 1:
                raise ValueError(
                    "Multiple partial decompositions of one token are not supported"
                )
            group = next(iter(reductions))
            if set(group.ranks) != ranks or any(r not in partials for r in ranks):
                raise ValueError(
                    "Partial contributions must cover exactly their reduction group"
                )
            if any((replicas[r] & ranks) - {r} for r in ranks):
                raise ValueError(
                    "Partial contributions cannot also be declared replicas"
                )
        else:
            connected = set()
            pending = {next(iter(ranks))}
            while pending:
                rank = pending.pop()
                connected.add(rank)
                pending.update((replicas[rank] & ranks) - connected)
            if connected != ranks:
                raise ValueError(
                    "Repeated tokens require an explicit Replicate relationship"
                )
