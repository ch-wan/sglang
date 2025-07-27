import torch

class PermuteMethodBase(ABC):
    
    @abstractmethod
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def unpermute(self, x: torch.Tensor) -> torch.Tensor:
        pass


class GroupedGEMMBaseWrapper(torch.nn.Module):
    pass
