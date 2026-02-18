import torch
import torch.nn as nn
from flashinfer.norm import rmsnorm, fused_add_rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual
        return rmsnorm(x, self.weight, self.eps)
