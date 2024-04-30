from __future__ import annotations

import torch


def normalize_params(params: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalized_params = torch.zeros_like(params)
    normalized_params[mask] = params[mask]
    normalized_params /= torch.sum(normalized_params, dim=-1, keepdim=True)
    assert (normalized_params[~mask] == 0.0).all()
    return normalized_params
