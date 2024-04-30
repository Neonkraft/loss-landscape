from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

from oneshot.archsampler import BaseSampler


class DRNASSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters, sample_frequency=sample_frequency
        )

    def sample_alphas(self, arch_parameters: torch.Tensor) -> list[torch.Tensor]:
        sampled_alphas = []
        for alpha in arch_parameters:
            sampled_alphas.append(self.sample(alpha))
        return sampled_alphas

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = F.elu(alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        return weights  # type: ignore
