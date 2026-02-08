from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBoundedIncrementalInductionConfiguration:
  pool_capacity: int = 8
  exploration_transformer_executions_per_step: int = 3
  validation_window_length: int = 256
  candidate_buffer_capacity: int = 32

  # To keep the demo reasonably fast, candidate validation is run every N steps
  # (instead of every step).
  candidate_validation_interval: int = 32

  # Freezing evaluation (incumbent evaluation on the validation window) is run every N steps.
  freeze_evaluation_interval: int = 32

  # Mixture smoothing (probability floor).
  probability_floor: float = 1e-3

  # Candidate acceptance: MDL(candidate) <= baseline_loss - detectability_slack_bits
  detectability_slack_bits: float = 8.0

  # Reacquisition metric tolerance: average regret <= tolerance_bits_per_character
  reacquisition_tolerance_bits_per_character: float = 0.25

  # Recall behavior
  maximum_recalled_programs_per_transformer: int = 4

  # Parallelism
  maximum_worker_threads: int | None = None
