from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.a2a_output import DispatchOutput

class BaseDispatcherConfig(ABC):
    """Base class for dispatcher configs."""
    pass


class BaseDispatcher(ABC):
    """Base class for dispatchers."""
    
    @abstractmethod
    def dispatch(self, *args, **kwargs) -> DispatchOutput:
        pass
    
    @abstractmethod
    def combine(self, *args, **kwargs):
        pass
