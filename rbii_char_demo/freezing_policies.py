from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class freezing_policy(Protocol):
    def should_freeze(
        self,
        time_step_index: int,
        incumbent_run_length: int,
        incumbent_average_loss_bits: float,
        baseline_average_loss_bits: float,
    ) -> bool: ...


@dataclass(frozen=True)
class always_freeze_incumbent_policy:
    def should_freeze(
        self,
        time_step_index: int,
        incumbent_run_length: int,
        incumbent_average_loss_bits: float,
        baseline_average_loss_bits: float,
    ) -> bool:
        return True


@dataclass(frozen=True)
class incumbent_run_length_freeze_policy:
    minimum_incumbent_run_length: int = 256
    minimum_average_gain_bits_per_character: float = 0.1

    def should_freeze(
        self,
        time_step_index: int,
        incumbent_run_length: int,
        incumbent_average_loss_bits: float,
        baseline_average_loss_bits: float,
    ) -> bool:
        if incumbent_run_length < self.minimum_incumbent_run_length:
            return False
        average_gain = baseline_average_loss_bits - incumbent_average_loss_bits
        return average_gain >= self.minimum_average_gain_bits_per_character
