from __future__ import annotations


class OneShotComponent:
    def __init__(self) -> None:
        self._epoch = 0
        self._step = 0

    def new_epoch(self) -> None:
        self._epoch += 1

    def new_step(self) -> None:
        self._step += 1
