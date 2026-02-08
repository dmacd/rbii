from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import math


class NewbornWeightAssignmentPolicy(Protocol):
  def compute_initial_log_weight_base_two(
      self,
      log_total_weight_base_two: float,
      transformer_description_length_bits: float,
  ) -> float: ...


@dataclass(frozen=True)
class PriorConsistentNewbornWeightAssignmentPolicy:
  def compute_initial_log_weight_base_two(
      self,
      log_total_weight_base_two: float,
      transformer_description_length_bits: float,
  ) -> float:
    # Mirrors Assumption 3 in the paper: W_new = 2^{-k} * sum(W_existing)
    return float(log_total_weight_base_two) - float(
      transformer_description_length_bits)


@dataclass(frozen=True)
class FixedFractionNewbornWeightAssignmentPolicy:
  fraction_of_total_weight: float = 0.01

  def compute_initial_log_weight_base_two(
      self,
      log_total_weight_base_two: float,
      transformer_description_length_bits: float,
  ) -> float:
    if self.fraction_of_total_weight <= 0.0 or self.fraction_of_total_weight >= 1.0:
      raise ValueError("fraction_of_total_weight must be in (0, 1).")
    return float(log_total_weight_base_two) + math.log2(
      self.fraction_of_total_weight)
