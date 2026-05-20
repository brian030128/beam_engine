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
        orig_shape = x.shape
        if residual is not None:
            x = x.reshape(-1, orig_shape[-1])
            residual = residual.reshape(-1, orig_shape[-1])
            fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x.view(orig_shape), residual.view(orig_shape)
        return rmsnorm(x.reshape(-1, orig_shape[-1]), self.weight, self.eps).view(orig_shape)
