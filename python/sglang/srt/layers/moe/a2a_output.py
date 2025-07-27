from enum import Enum, auto
from typing import NamedTuple, Protocol, runtime_checkable


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
    pass


class DeepEPNormalOutput(NamedTuple):
    """DeepEP normal dispatch output."""
    pass


class DeepEPLLOutput(NamedTuple):
    """DeepEP low latency dispatch output."""
    pass


assert issubclass(StandardDispatchOutput, DispatchOutput)
assert issubclass(DeepEPNormalOutput, DispatchOutput)
assert issubclass(DeepEPLLOutput, DispatchOutput)
