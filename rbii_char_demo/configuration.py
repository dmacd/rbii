from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBoundedIncrementalInductionConfiguration:
  pool_capacity: int = 8

  # Exploration: how many candidate transformers/predictors to validate per step.
  exploration_transformer_executions_per_step: int = 3

  # Transformer search (DreamCoder-style enumerative search)
  transformer_search_probability_budget_bits: int = 20
  transformer_search_maximum_expression_depth: int = 6
  transformer_search_maximum_expressions_per_step: int = 64

  validation_window_length: int = 256
  candidate_buffer_capacity: int = 32

  candidate_validation_interval: int = 32
  freeze_evaluation_interval: int = 32

  probability_floor: float = 1e-3
  detectability_slack_bits: float = 8.0
  reacquisition_tolerance_bits_per_character: float = 0.25

  maximum_recalled_programs_per_step: int = 4

  maximum_worker_threads: int | None = None
