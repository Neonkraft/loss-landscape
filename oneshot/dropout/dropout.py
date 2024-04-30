# from __future__ import annotations

# import math
# from typing import Literal

# import torch

# from oneshot.base_component import OneShotComponent


# class Dropout(OneShotComponent):
#     """A class representing a dropout operation for architectural parameters."""

#     def __init__(
#         self,
#         p: float,
#         p_min: float | None = None,
#         anneal_frequency: Literal["epoch", "step", None] = None,
#         anneal_type: Literal["linear", "cosine", None] = None,
#         max_iter: int | None = None,
#     ) -> None:
#         """Instantiate a dropout class.

#         Args:
#             p (float): The initial dropout probability, must be in the range [0, 1).
#             p_min (float, optional): The dropout probability to decay to.
#             Must be in the range [0, 1)
#             anneal_frequency (str, optional): The frequency at which to anneal the
#             dropout probability. Defaults to None.
#             anneal_type (str, optional): Type of probability annealing to be used.
#             Defaults to None.
#             max_iter (int, optional): Total amount of iterations.
#             Required for annealing.
#         """
#         super().__init__()
#         assert p >= 0
#         assert p < 1
#         assert anneal_frequency in ["epoch", "step", None]
#         assert anneal_type in ["linear", "cosine", None]
#         assert bool(anneal_frequency) == bool(anneal_type)
#         if anneal_frequency is not None:
#             assert p_min >= 0  # type: ignore
#             assert p_min < 1  # type: ignore
#             assert p_min < p  # type: ignore
#             assert max_iter > 0  # type: ignore

#         self._p_init = p
#         self._p_min = p_min
#         self._anneal_frequency = anneal_frequency
#         self._anneal_type = anneal_type
#         self._max_iter = max_iter

#         self.p = self._p_init

#     def apply_mask(self, parameters: torch.Tensor) -> torch.Tensor:
#         r"""This function masks the parameters based on the drop probability p.
#         Additionally, the values are scaled by the factor of :math:`\frac{1}{1-p}`
#         in order to ensure that during evaluation the module simply computes an
#         identity function.
#         """
#         random = torch.rand_like(parameters)
#         dropout_mask = random >= self.p
#         rescale = 1 / (1 - self.p)
#         return rescale * dropout_mask * parameters

#     def new_epoch(self) -> None:
#         super().new_epoch()
#         if self._anneal_frequency == "epoch":
#             self._anneal_probability()

#     def new_step(self) -> None:
#         super().new_step()
#         if self._anneal_frequency == "step":
#             self._anneal_probability()

#     def _anneal_probability(self) -> None:
#         """This function decays the dropout probability to the goal probability."""
#         if self._anneal_type == "linear":
#             self.p = (1 - self._epoch / self._max_iter) * (  # type: ignore
#                 self._p_init - self._p_min  # type: ignore
#             ) + self._p_min  # type: ignore
#         elif self._anneal_type == "cosine":
#             # Concept from the following paper:
#             # Loshchilov, I., & Hutter, F. (2016).
#             # SGDR: Stochastic gradient descent with warm restarts.
#             self.p = (
#                 0.5  # type: ignore
#                 * (self._p_init - self._p_min)  # type: ignore
#                 * (1 + math.cos(self.p * math.pi / self._max_iter))  # type: ignore
#                 + self._p_min
#             )
#         else:
#             raise ValueError(f"Unsupported annealing type: {self._anneal_type}")
