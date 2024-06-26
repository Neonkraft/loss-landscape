from __future__ import annotations

import torch
from torch import nn

# from oneshot.dropout import Dropout
from oneshot.partial_connector import PartialConnector
from oneshot.weightentangler import WeightEntangler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
__all__ = ["OperationChoices"]


class OperationChoices(nn.Module):
    def __init__(self, ops: list[nn.Module], is_reduction_cell: bool = False) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]
        return sum(states)  # type: ignore

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None or wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=DEVICE)  # type: ignore


class OperationBlock(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool,
        partial_connector: PartialConnector | None = None,
        dropout: str | None = None,
        weight_entangler: WeightEntangler | None = None,
        device: torch.device = DEVICE,
        is_argmax_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        if partial_connector:
            for op in ops:
                if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                    op.change_channel_size(
                        partial_connector.k, self.device  # type: ignore
                    )
        self.ops = ops
        self.partial_connector = partial_connector
        self.is_reduction_cell = is_reduction_cell
        self.dropout = dropout
        self.weight_entangler = weight_entangler
        self.is_argmax_sampler = is_argmax_sampler

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)

        # TODO[DESIGN]: How can weight entanglement work with partial connections?
        if self.weight_entangler is not None:
            return self.weight_entangler.forward(x, self.ops, alphas)

        # TODO[DESIGN]: PartialConnector shouldn't have to implement everything that
        # follows this if block, but it does. Not very clean.
        if self.partial_connector:
            self.partial_connector.is_reduction_cell = self.is_reduction_cell
            return self.partial_connector(x, alphas, self.ops, self.is_argmax_sampler)

        if self.is_argmax_sampler:
            argmax = torch.argmax(alphas)
            states = [
                alphas[i] * op(x) if i == argmax else alphas[i]
                for i, op in enumerate(self.ops)
            ]
        else:
            states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]

        output = sum(states)
        return output

    def change_op_channel_size(self, wider: int | None = None) -> None:
        if wider is None:
            wider = self.partial_connector.k if self.partial_connector else 1
        if wider == 1:
            return

        for op in self.ops:
            if not (isinstance(op, (nn.AvgPool2d, nn.MaxPool2d))):
                op.change_channel_size(k=1 / wider, device=self.device)  # type: ignore
