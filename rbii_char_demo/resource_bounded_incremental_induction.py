from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy

from configuration import ResourceBoundedIncrementalInductionConfiguration
from domain_specific_language import DomainSpecificLanguageEvaluationContext, \
  PrimitiveCallExpression
from memory_mechanisms import MemoryMechanism, \
  character_history_memory_mechanism
from predictor_programs import DomainSpecificLanguagePredictorProgram, \
  PredictorProgram
from primitive_library import PrimitiveLibrary, create_default_primitive_library
from transformer_programs import (
  EnumerativeTransformerSearchStrategy,
  ProbabilisticProgramGrammar,
  TransformerSearchStrategy,
  create_default_probabilistic_program_grammar,
)


class FreezingPolicy(Protocol):
  def should_freeze(
      self,
      time_step_index: int,
      incumbent_program_identifier: str,
      incumbent_run_length: int,
      validation_buffer_length: int,
  ) -> bool: ...


class NewbornWeightAssignmentPolicy(Protocol):
  def compute_newborn_log_weight_base_two(
      self,
      description_length_bits: float,
      current_log_total_weight_base_two: float,
  ) -> float: ...


class AlwaysFreezeIncumbentPolicy:
  def should_freeze(
      self,
      time_step_index: int,
      incumbent_program_identifier: str,
      incumbent_run_length: int,
      validation_buffer_length: int,
  ) -> bool:
    del time_step_index
    del incumbent_program_identifier
    del incumbent_run_length
    del validation_buffer_length
    return True


class PriorConsistentNewbornWeightAssignmentPolicy:
  """Implements Assumption 3-style weight injection: W_new = 2^{-k} * Z."""

  def compute_newborn_log_weight_base_two(
      self,
      description_length_bits: float,
      current_log_total_weight_base_two: float,
  ) -> float:
    return float(current_log_total_weight_base_two) - float(
      description_length_bits)


@dataclass(frozen=True)
class ValidationBufferEntry:
  evaluation_features: Any
  observed_character_index: int
  recall_key: str


@dataclass
class ActivePredictorEntry:
  program_identifier: str
  predictor_program: PredictorProgram
  description_length_bits: float
  log_weight_base_two: float


@dataclass
class CandidatePredictorEntry:
  program_identifier: str
  predictor_program: PredictorProgram
  description_length_bits: float
  minimum_description_length_score_bits: float


class FrozenProgramStore:
  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._programs_by_identifier: dict[str, PredictorProgram] = {}

  def add_program(self, program_identifier: str,
                  predictor_program: PredictorProgram) -> None:
    with self._lock:
      self._programs_by_identifier[str(program_identifier)] = predictor_program

  def has_program(self, program_identifier: str) -> bool:
    with self._lock:
      return str(program_identifier) in self._programs_by_identifier

  def get_program(self, program_identifier: str) -> PredictorProgram:
    with self._lock:
      return self._programs_by_identifier[str(program_identifier)]

  def list_program_identifiers(self) -> list[str]:
    with self._lock:
      return list(self._programs_by_identifier.keys())


class ResourceBoundedIncrementalInduction:
  """Resource Bounded Incremental Induction (RBII) loop for character prediction.

  Key implementation points:
      - Active pool is finite and tracked by exp-weights (log weights).
      - Exploration proposes DSL transformer expressions via a DreamCoder-style enumerator.
      - Candidates are validated on a fixed recent window via MDL: loss + description_length_bits.
      - Frozen store is indexed by the memory mechanism under a recall key (state-indexed recall).
  """

  def __init__(
      self,
      character_vocabulary: Any,
      configuration: ResourceBoundedIncrementalInductionConfiguration | None = None,
      memory_mechanism: MemoryMechanism | None = None,
      primitive_library: PrimitiveLibrary | None = None,
      transformer_search_strategy: TransformerSearchStrategy | None = None,
      freezing_policy: FreezingPolicy | None = None,
      newborn_weight_assignment_policy: NewbornWeightAssignmentPolicy | None = None,
  ) -> None:
    self.character_vocabulary = character_vocabulary
    self.configuration = configuration or ResourceBoundedIncrementalInductionConfiguration()

    self.memory_mechanism = memory_mechanism or character_history_memory_mechanism(
      maximum_context_length=16,
      maximum_history_length=2048,
      recall_key_length=4,
      maximum_program_identifiers_per_key=8,
    )
    self.memory_state = self.memory_mechanism.initialize(
      character_vocabulary=self.character_vocabulary)

    self.primitive_library = primitive_library or create_default_primitive_library()

    if transformer_search_strategy is None:
      grammar = create_default_probabilistic_program_grammar(
        primitive_library=self.primitive_library)
      transformer_search_strategy = EnumerativeTransformerSearchStrategy(
        grammar=grammar,
        probability_budget_bits=float(
          self.configuration.transformer_search_probability_budget_bits),
        maximum_expression_depth=int(
          self.configuration.transformer_search_maximum_expression_depth),
      )
    self.transformer_search_strategy = transformer_search_strategy

    self.freezing_policy = freezing_policy or AlwaysFreezeIncumbentPolicy()
    self.newborn_weight_assignment_policy = newborn_weight_assignment_policy or PriorConsistentNewbornWeightAssignmentPolicy()

    self.frozen_store = FrozenProgramStore()

    self.active_pool: list[ActivePredictorEntry] = []
    self.candidate_buffer: list[CandidatePredictorEntry] = []
    self.validation_buffer: list[ValidationBufferEntry] = []

    self.time_step_index = 0
    self.incumbent_program_identifier = ""
    self.incumbent_run_length = 0

    self._thread_pool_executor = ThreadPoolExecutor(
      max_workers=self.configuration.maximum_worker_threads)
    self._lock = threading.Lock()

    self._initialize_with_uniform_predictor()

    self.last_mixture_distribution: numpy.ndarray | None = None
    self.last_loss_bits: float | None = None

  def _initialize_with_uniform_predictor(self) -> None:
    uniform_expression = PrimitiveCallExpression(
      primitive_name="uniform_character_distribution", argument_expressions=())
    uniform_predictor = DomainSpecificLanguagePredictorProgram(
      expression=uniform_expression)
    program_identifier = self._compute_program_identifier(uniform_expression)
    self.active_pool.append(
      ActivePredictorEntry(
        program_identifier=program_identifier,
        predictor_program=uniform_predictor,
        description_length_bits=0.0,
        log_weight_base_two=0.0,
      )
    )
    self.incumbent_program_identifier = program_identifier
    self.incumbent_run_length = 0

  def _compute_program_identifier(self, expression: Any) -> str:
    encoded = repr(expression).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

  def step(self, observed_character_index: int) -> numpy.ndarray:
    """Processes one observation.

    Returns:
        The mixture distribution predicted *before* seeing the observation.
    """

    observed_index = int(observed_character_index)

    # Build evaluation context for the current state (predicting x_t given m_{t-1}).
    prediction_features = self.memory_mechanism.build_prediction_features(
      memory_state=self.memory_state,
      probability_floor=float(self.configuration.probability_floor),
    )
    recall_key = self.memory_mechanism.build_recall_key(
      memory_state=self.memory_state)
    recalled_program_identifiers = tuple(
      self.memory_mechanism.recall_program_identifiers(
        memory_state=self.memory_state,
        recall_key=recall_key,
        maximum_number_of_program_identifiers=int(
          self.configuration.maximum_recalled_programs_per_step),
      )
    )

    evaluation_context = DomainSpecificLanguageEvaluationContext(
      character_vocabulary=self.character_vocabulary,
      prediction_features=prediction_features,
      recalled_frozen_program_identifiers=recalled_program_identifiers,
      probability_floor=float(self.configuration.probability_floor),
    )

    # Predict with all active predictors (parallelizable in no-GIL python).
    predictor_distributions = self._predict_active_pool_distributions(
      evaluation_context=evaluation_context)

    mixture_distribution = self._compute_mixture_distribution(
      predictor_distributions=predictor_distributions)
    mixture_distribution = self._apply_probability_floor(mixture_distribution)

    self.last_mixture_distribution = mixture_distribution
    self.last_loss_bits = self._negative_log_probability_bits(
      mixture_distribution, observed_index)

    # Update weights.
    self._update_active_pool_weights(
      predictor_distributions=predictor_distributions,
      observed_character_index=observed_index,
    )

    # Update state with the observation.
    self.memory_mechanism.update(memory_state=self.memory_state,
                                 observed_character_index=observed_index)

    # Append to validation buffer (store features from m_{t-1} with the observed x_t).
    self.validation_buffer.append(
      ValidationBufferEntry(
        evaluation_features=prediction_features,
        observed_character_index=observed_index,
        recall_key=str(recall_key),
      )
    )
    if len(self.validation_buffer) > int(
        self.configuration.validation_window_length):
      self.validation_buffer = self.validation_buffer[
        -int(self.configuration.validation_window_length):]

    self.time_step_index += 1

    # Periodic candidate re-validation.
    if int(self.configuration.candidate_validation_interval) > 0 and (
        self.time_step_index % int(
      self.configuration.candidate_validation_interval) == 0
    ):
      self._revalidate_candidate_buffer()

    # Explore: propose and validate new transformer expressions.
    self._explore_and_maybe_insert_candidates(
      evaluation_context=evaluation_context)

    # Freeze: store the incumbent under the current recall key (state-indexed recall).
    if int(self.configuration.freeze_evaluation_interval) > 0 and (
        self.time_step_index % int(
      self.configuration.freeze_evaluation_interval) == 0
    ):
      self._maybe_freeze_incumbent()

    return mixture_distribution

  def _predict_active_pool_distributions(
      self, evaluation_context: DomainSpecificLanguageEvaluationContext
  ) -> list[numpy.ndarray]:
    def compute_distribution(entry: ActivePredictorEntry) -> numpy.ndarray:
      try:
        distribution = entry.predictor_program.predict_character_distribution(
          evaluation_context=evaluation_context,
          primitive_library=self.primitive_library,
          frozen_store=self.frozen_store,
        )
        return numpy.asarray(distribution, dtype=numpy.float64)
      except Exception:
        vocabulary_size = int(self.character_vocabulary.size)
        return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
          vocabulary_size)

    return list(
      self._thread_pool_executor.map(compute_distribution, self.active_pool))

  def _compute_mixture_distribution(self, predictor_distributions: list[
    numpy.ndarray]) -> numpy.ndarray:
    vocabulary_size = int(self.character_vocabulary.size)
    if len(predictor_distributions) == 0:
      return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)

    log_weights = numpy.array(
      [float(entry.log_weight_base_two) for entry in self.active_pool],
      dtype=numpy.float64)
    maximum_log_weight = float(log_weights.max())
    stabilized_weights = numpy.power(2.0, log_weights - maximum_log_weight)
    total_weight = float(stabilized_weights.sum())
    if not numpy.isfinite(total_weight) or total_weight <= 0.0:
      stabilized_weights = numpy.ones_like(stabilized_weights,
                                           dtype=numpy.float64)
      total_weight = float(stabilized_weights.sum())

    normalized_weights = stabilized_weights / total_weight

    mixture = numpy.zeros(vocabulary_size, dtype=numpy.float64)
    for weight, distribution in zip(normalized_weights,
                                    predictor_distributions):
      distribution_array = numpy.asarray(distribution, dtype=numpy.float64)
      if distribution_array.shape[0] != vocabulary_size:
        continue
      mixture += float(weight) * distribution_array

    total = float(mixture.sum())
    if not numpy.isfinite(total) or total <= 0.0:
      return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)
    return mixture / total

  def _apply_probability_floor(self,
                               distribution: numpy.ndarray) -> numpy.ndarray:
    vocabulary_size = int(self.character_vocabulary.size)
    probability_floor = float(self.configuration.probability_floor)
    distribution_array = numpy.asarray(distribution, dtype=numpy.float64)
    if distribution_array.shape[0] != vocabulary_size:
      distribution_array = numpy.ones(vocabulary_size,
                                      dtype=numpy.float64) / float(
        vocabulary_size)

    if probability_floor <= 0.0:
      total = float(distribution_array.sum())
      if total > 0.0:
        return distribution_array / total
      return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)

    floored = numpy.maximum(distribution_array, probability_floor)
    total = float(floored.sum())
    if total <= 0.0 or not numpy.isfinite(total):
      return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)
    return floored / total

  def _negative_log_probability_bits(self, distribution: numpy.ndarray,
                                     observed_character_index: int) -> float:
    vocabulary_size = int(self.character_vocabulary.size)
    index = int(observed_character_index)
    if index < 0 or index >= vocabulary_size:
      return float(math.log2(vocabulary_size))

    probability_value = float(distribution[index])
    probability_floor = float(self.configuration.probability_floor)
    probability_value = max(probability_value,
                            probability_floor if probability_floor > 0.0 else 1e-12)
    return float(-math.log2(probability_value))

  def _update_active_pool_weights(
      self,
      predictor_distributions: list[numpy.ndarray],
      observed_character_index: int,
  ) -> None:
    index = int(observed_character_index)
    vocabulary_size = int(self.character_vocabulary.size)
    probability_floor = float(self.configuration.probability_floor)
    probability_floor = probability_floor if probability_floor > 0.0 else 1e-12

    for entry, distribution in zip(self.active_pool, predictor_distributions):
      if index < 0 or index >= vocabulary_size:
        probability_value = 1.0 / float(vocabulary_size)
      else:
        probability_value = float(distribution[index]) if float(
          distribution[index]) > 0.0 else probability_floor
        probability_value = max(probability_value, probability_floor)
      entry.log_weight_base_two += float(math.log2(probability_value))

    # Re-center weights occasionally to avoid drift.
    log_weights = [float(entry.log_weight_base_two) for entry in
                   self.active_pool]
    if len(log_weights) > 0:
      maximum_log_weight = max(log_weights)
      for entry in self.active_pool:
        entry.log_weight_base_two -= maximum_log_weight

  def _compute_current_log_total_weight_base_two(self) -> float:
    log_weights = numpy.array(
      [float(entry.log_weight_base_two) for entry in self.active_pool],
      dtype=numpy.float64)
    maximum_log_weight = float(log_weights.max())
    stabilized_weights = numpy.power(2.0, log_weights - maximum_log_weight)
    total_weight = float(stabilized_weights.sum())
    if not numpy.isfinite(total_weight) or total_weight <= 0.0:
      return 0.0
    return float(math.log2(total_weight)) + maximum_log_weight

  def _explore_and_maybe_insert_candidates(self,
                                           evaluation_context: DomainSpecificLanguageEvaluationContext) -> None:
    maximum_number_of_transformers = int(
      self.configuration.exploration_transformer_executions_per_step)
    if maximum_number_of_transformers <= 0:
      return

    candidates = self.transformer_search_strategy.propose_transformer_expressions(
      evaluation_context=evaluation_context,
      maximum_number_of_expressions=maximum_number_of_transformers,
    )

    for candidate in candidates:
      expression = candidate.transformer_expression
      program_identifier = self._compute_program_identifier(expression)

      if any(entry.program_identifier == program_identifier for entry in
             self.active_pool):
        continue
      if any(entry.program_identifier == program_identifier for entry in
             self.candidate_buffer):
        continue

      predictor_program = DomainSpecificLanguagePredictorProgram(
        expression=expression)
      minimum_description_length_score_bits = self._score_candidate_minimum_description_length(
        predictor_program=predictor_program,
        description_length_bits=float(candidate.description_length_bits),
      )

      self.candidate_buffer.append(
        CandidatePredictorEntry(
          program_identifier=program_identifier,
          predictor_program=predictor_program,
          description_length_bits=float(candidate.description_length_bits),
          minimum_description_length_score_bits=float(
            minimum_description_length_score_bits),
        )
      )

    self._trim_candidate_buffer()
    self._insert_detectable_candidates()

  def _score_candidate_minimum_description_length(
      self,
      predictor_program: PredictorProgram,
      description_length_bits: float,
  ) -> float:
    window_loss_bits = self._compute_validation_window_loss_bits(
      predictor_program=predictor_program)

    vocabulary_size = int(self.character_vocabulary.size)
    uniform_loss_bits_per_character = float(math.log2(vocabulary_size))
    baseline_loss_bits = float(
      len(self.validation_buffer)) * uniform_loss_bits_per_character

    return float(window_loss_bits) + float(description_length_bits)

  def _compute_validation_window_loss_bits(self,
                                           predictor_program: PredictorProgram) -> float:
    if len(self.validation_buffer) == 0:
      return 0.0

    vocabulary_size = int(self.character_vocabulary.size)

    recalled_program_identifiers_by_key: dict[str, tuple[str, ...]] = {}
    for entry in self.validation_buffer:
      if entry.recall_key in recalled_program_identifiers_by_key:
        continue
      recalled_program_identifiers_by_key[entry.recall_key] = tuple(
        self.memory_mechanism.recall_program_identifiers(
          memory_state=self.memory_state,
          recall_key=entry.recall_key,
          maximum_number_of_program_identifiers=int(
            self.configuration.maximum_recalled_programs_per_step),
        )
      )

    total_loss_bits = 0.0
    for entry in self.validation_buffer:
      recalled_program_identifiers = recalled_program_identifiers_by_key.get(
        entry.recall_key, ())
      evaluation_context = DomainSpecificLanguageEvaluationContext(
        character_vocabulary=self.character_vocabulary,
        prediction_features=entry.evaluation_features,
        recalled_frozen_program_identifiers=recalled_program_identifiers,
        probability_floor=float(self.configuration.probability_floor),
      )
      try:
        distribution = predictor_program.predict_character_distribution(
          evaluation_context=evaluation_context,
          primitive_library=self.primitive_library,
          frozen_store=self.frozen_store,
        )
        distribution_array = numpy.asarray(distribution, dtype=numpy.float64)
      except Exception:
        distribution_array = numpy.ones(vocabulary_size,
                                        dtype=numpy.float64) / float(
          vocabulary_size)

      distribution_array = self._apply_probability_floor(distribution_array)
      total_loss_bits += self._negative_log_probability_bits(distribution_array,
                                                             int(
                                                               entry.observed_character_index))

    return float(total_loss_bits)

  def _trim_candidate_buffer(self) -> None:
    capacity = int(self.configuration.candidate_buffer_capacity)
    if capacity <= 0:
      self.candidate_buffer = []
      return
    if len(self.candidate_buffer) <= capacity:
      return
    self.candidate_buffer.sort(
      key=lambda entry: float(entry.minimum_description_length_score_bits))
    self.candidate_buffer = self.candidate_buffer[:capacity]

  def _insert_detectable_candidates(self) -> None:
    if len(self.validation_buffer) == 0:
      return

    vocabulary_size = int(self.character_vocabulary.size)
    uniform_loss_bits_per_character = float(math.log2(vocabulary_size))
    baseline_loss_bits = float(
      len(self.validation_buffer)) * uniform_loss_bits_per_character
    slack_bits = float(self.configuration.detectability_slack_bits)

    detectable_entries = [
      entry
      for entry in self.candidate_buffer
      if float(entry.minimum_description_length_score_bits) <= float(
        baseline_loss_bits - slack_bits)
    ]

    if len(detectable_entries) == 0:
      return

    detectable_entries.sort(
      key=lambda entry: float(entry.minimum_description_length_score_bits))
    for entry in detectable_entries:
      self._insert_into_active_pool(entry)

    self.candidate_buffer = [
      entry for entry in self.candidate_buffer if
      entry.program_identifier not in {d.program_identifier for d in
                                       detectable_entries}
    ]

  def _insert_into_active_pool(self,
                               candidate_entry: CandidatePredictorEntry) -> None:
    current_log_total_weight_base_two = self._compute_current_log_total_weight_base_two()
    newborn_log_weight_base_two = self.newborn_weight_assignment_policy.compute_newborn_log_weight_base_two(
      description_length_bits=float(candidate_entry.description_length_bits),
      current_log_total_weight_base_two=float(
        current_log_total_weight_base_two),
    )

    self.active_pool.append(
      ActivePredictorEntry(
        program_identifier=str(candidate_entry.program_identifier),
        predictor_program=candidate_entry.predictor_program,
        description_length_bits=float(candidate_entry.description_length_bits),
        log_weight_base_two=float(newborn_log_weight_base_two),
      )
    )

    self._enforce_pool_capacity()
    self._update_incumbent_tracking()

  def _enforce_pool_capacity(self) -> None:
    capacity = int(self.configuration.pool_capacity)
    if capacity <= 0:
      capacity = 1
    if len(self.active_pool) <= capacity:
      return

    self._update_incumbent_tracking()
    incumbent_identifier = str(self.incumbent_program_identifier)

    self.active_pool.sort(key=lambda entry: float(entry.log_weight_base_two))
    pruned: list[ActivePredictorEntry] = []
    for entry in self.active_pool:
      if len(pruned) >= len(self.active_pool) - capacity:
        break
      if entry.program_identifier == incumbent_identifier:
        continue
      pruned.append(entry)

    pruned_identifiers = {entry.program_identifier for entry in pruned}
    self.active_pool = [entry for entry in self.active_pool if
                        entry.program_identifier not in pruned_identifiers]

  def _update_incumbent_tracking(self) -> None:
    if len(self.active_pool) == 0:
      self.incumbent_program_identifier = ""
      self.incumbent_run_length = 0
      return

    best_entry = max(self.active_pool,
                     key=lambda entry: float(entry.log_weight_base_two))
    new_identifier = str(best_entry.program_identifier)

    if new_identifier == self.incumbent_program_identifier:
      self.incumbent_run_length += 1
    else:
      self.incumbent_program_identifier = new_identifier
      self.incumbent_run_length = 1

  def _revalidate_candidate_buffer(self) -> None:
    if len(self.candidate_buffer) == 0:
      return

    def rescore(entry: CandidatePredictorEntry) -> CandidatePredictorEntry:
      score_bits = self._score_candidate_minimum_description_length(
        predictor_program=entry.predictor_program,
        description_length_bits=float(entry.description_length_bits),
      )
      return CandidatePredictorEntry(
        program_identifier=entry.program_identifier,
        predictor_program=entry.predictor_program,
        description_length_bits=float(entry.description_length_bits),
        minimum_description_length_score_bits=float(score_bits),
      )

    rescored = list(
      self._thread_pool_executor.map(rescore, self.candidate_buffer))
    self.candidate_buffer = rescored
    self._trim_candidate_buffer()
    self._insert_detectable_candidates()

  def _maybe_freeze_incumbent(self) -> None:
    if self.incumbent_program_identifier == "":
      return

    should_freeze = self.freezing_policy.should_freeze(
      time_step_index=int(self.time_step_index),
      incumbent_program_identifier=str(self.incumbent_program_identifier),
      incumbent_run_length=int(self.incumbent_run_length),
      validation_buffer_length=int(len(self.validation_buffer)),
    )
    if not should_freeze:
      return

    incumbent_entry = next(
      (entry for entry in self.active_pool if
       entry.program_identifier == self.incumbent_program_identifier),
      None,
    )
    if incumbent_entry is None:
      return

    self.frozen_store.add_program(
      program_identifier=incumbent_entry.program_identifier,
      predictor_program=incumbent_entry.predictor_program)

    current_recall_key = self.memory_mechanism.build_recall_key(
      memory_state=self.memory_state)
    self.memory_mechanism.record_frozen_program_identifier(
      memory_state=self.memory_state,
      recall_key=str(current_recall_key),
      frozen_program_identifier=str(incumbent_entry.program_identifier),
    )
