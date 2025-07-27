from enum import Enum, auto
from typing import List, NamedTuple, Optional, Protocol, Tuple, runtime_checkable

import torch


class DispatchOutputFormat(Enum):
    standard = auto()
    deepep_normal = auto()
    deepep_ll = auto()
    
    def is_standard(self) -> bool:
        return self == DispatchOutputFormat.standard
    
    def is_deepep_normal(self) -> bool:
        return self == DispatchOutputFormat.deepep_normal
    
    def is_deepep_ll(self) -> bool:
        return self == DispatchOutputFormat.deepep_ll
    

@runtime_checkable
class DispatchOutput(Protocol):
    """Protocol for dispatch outputs in different formats."""
    @property
    def format(self) -> DispatchOutputFormat:
        ...


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.standard


class DeepEPNormalOutput(NamedTuple):
    """DeepEP normal dispatch output."""
    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    topk_idx: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]
    num_recv_tokens_per_expert_list: List[int]
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.deepep_normal


class DeepEPLLOutput(NamedTuple):
    """DeepEP low latency dispatch output."""
    hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor]
    masked_m: torch.Tensor
    expected_m: int
    
    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.deepep_ll


assert issubclass(StandardDispatchOutput, DispatchOutput)
assert issubclass(DeepEPNormalOutput, DispatchOutput)
assert issubclass(DeepEPLLOutput, DispatchOutput)
