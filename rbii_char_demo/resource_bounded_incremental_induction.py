from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as numpy

from character_vocabulary import CharacterVocabulary
from configuration import ResourceBoundedIncrementalInductionConfiguration
from domain_specific_language import (
    DomainSpecificLanguageEvaluationContext,
    PredictionFeatures,
    PrimitiveCallExpression,
)
from freezing_policies import FreezingPolicy
from memory_mechanisms import MemoryMechanism
from newborn_weight_policies import NewbornWeightAssignmentPolicy
from predictor_programs import DomainSpecificLanguagePredictorProgram
from primitive_library import PrimitiveLibrary
from transformer_programs import TransformerProgram, TransformerCandidate


@dataclass
class FrozenProgramRecord:
    program_identifier: str
    predictor_program: DomainSpecificLanguagePredictorProgram
    transformer_description_length_bits: int
    times_recalled: int = 0
    times_frozen: int = 1


class FrozenProgramStore:
    def __init__(self) -> None:
        self._records_by_identifier: dict[str, FrozenProgramRecord] = {}
        self._program_identifiers_in_insertion_order: list[str] = []
        self._next_identifier_index: int = 0

    def add_program(
        self,
        predictor_program: DomainSpecificLanguagePredictorProgram,
        transformer_description_length_bits: int,
    ) -> str:
        program_identifier = f"frozen_program_{self._next_identifier_index}"
        self._next_identifier_index += 1
        record = FrozenProgramRecord(
            program_identifier=program_identifier,
            predictor_program=predictor_program,
            transformer_description_length_bits=int(transformer_description_length_bits),
        )
        self._records_by_identifier[program_identifier] = record
        self._program_identifiers_in_insertion_order.append(program_identifier)
        return program_identifier

    def get_program(self, program_identifier: str) -> DomainSpecificLanguagePredictorProgram:
        return self._records_by_identifier[program_identifier].predictor_program

    def list_program_identifiers(self) -> list[str]:
        return list(self._program_identifiers_in_insertion_order)

    def record_recall(self, program_identifier: str) -> None:
        if program_identifier in self._records_by_identifier:
            self._records_by_identifier[program_identifier].times_recalled += 1

    @property
    def size(self) -> int:
        return len(self._program_identifiers_in_insertion_order)


@dataclass
class ActivePoolEntry:
    predictor_program: DomainSpecificLanguagePredictorProgram
    predictor_signature: str
    origin_transformer_description_length_bits: int
    logarithmic_weight_base_two: float
    instance_identifier: str


@dataclass(frozen=True)
class ValidationExample:
    prediction_features: PredictionFeatures
    observed_character_index: int


@dataclass
class ResourceBoundedIncrementalInductionRunResult:
    character_vocabulary: CharacterVocabulary
    per_step_algorithm_loss_bits: list[float]
    per_step_baseline_loss_bits: list[float]
    per_step_incumbent_signature: list[str]
    active_pool_size_over_time: list[int]
    frozen_store_size_over_time: list[int]


def _logarithmic_sum_base_two(logarithmic_values_base_two: Iterable[float]) -> float:
    values = list(logarithmic_values_base_two)
    if len(values) == 0:
        return -math.inf
    maximum_value = max(values)
    if not math.isfinite(maximum_value):
        return -math.inf
    sum_value = 0.0
    for value in values:
        sum_value += 2.0 ** (value - maximum_value)
    return maximum_value + math.log2(sum_value)


def _normalize_logarithmic_weights_base_two(logarithmic_weights_base_two: list[float]) -> list[float]:
    logarithmic_total_weight_base_two = _logarithmic_sum_base_two(logarithmic_weights_base_two)
    if not math.isfinite(logarithmic_total_weight_base_two):
        raise ValueError("Cannot normalize weights: total weight is non-finite.")
    return [value - logarithmic_total_weight_base_two for value in logarithmic_weights_base_two]


class ResourceBoundedIncrementalInductionSystem:
    def __init__(
        self,
        configuration: ResourceBoundedIncrementalInductionConfiguration,
        primitive_library: PrimitiveLibrary,
        transformer_programs: list[TransformerProgram],
        memory_mechanism: MemoryMechanism,
        freezing_policy: FreezingPolicy,
        newborn_weight_assignment_policy: NewbornWeightAssignmentPolicy,
        random_seed: int = 0,
    ) -> None:
        self.configuration = configuration
        self.primitive_library = primitive_library
        self.transformer_programs = transformer_programs
        self.memory_mechanism = memory_mechanism
        self.freezing_policy = freezing_policy
        self.newborn_weight_assignment_policy = newborn_weight_assignment_policy
        self.random_generator = random.Random(int(random_seed))

        transformer_weights = []
        for transformer in transformer_programs:
            transformer_weights.append(2.0 ** (-float(transformer.description_length_bits)))
        total_transformer_weight = float(sum(transformer_weights))
        self.transformer_sampling_probabilities = [weight / total_transformer_weight for weight in transformer_weights]

    def _sample_transformers_for_exploration(self) -> list[TransformerProgram]:
        number_to_sample = int(self.configuration.exploration_transformer_executions_per_step)
        return self.random_generator.choices(
            population=self.transformer_programs,
            weights=self.transformer_sampling_probabilities,
            k=number_to_sample,
        )

    def _create_thread_pool_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self.configuration.maximum_worker_threads)

    def run(self, character_indices: list[int], character_vocabulary: CharacterVocabulary) -> ResourceBoundedIncrementalInductionRunResult:
        frozen_store = FrozenProgramStore()
        memory_state = self.memory_mechanism.initialize(character_vocabulary)

        initial_expression = PrimitiveCallExpression("uniform_character_distribution", ())
        initial_program = DomainSpecificLanguagePredictorProgram(expression=initial_expression)

        active_pool: list[ActivePoolEntry] = [
            ActivePoolEntry(
                predictor_program=initial_program,
                predictor_signature=repr(initial_program.expression),
                origin_transformer_description_length_bits=2,
                logarithmic_weight_base_two=0.0,
                instance_identifier="active_instance_0",
            )
        ]
        next_active_instance_identifier_index = 1

        validation_window: list[ValidationExample] = []
        candidate_buffer_by_signature: dict[str, TransformerCandidate] = {}

        baseline_loss_per_character = math.log2(float(character_vocabulary.size))
        incumbent_signature = active_pool[0].predictor_signature
        incumbent_run_length = 0

        per_step_algorithm_loss_bits: list[float] = []
        per_step_baseline_loss_bits: list[float] = []
        per_step_incumbent_signature: list[str] = []
        active_pool_size_over_time: list[int] = []
        frozen_store_size_over_time: list[int] = []

        with self._create_thread_pool_executor() as executor:
            for time_step_index, observed_character_index in enumerate(character_indices):
                # ---- Build features for predictors (state at time t-1) ----
                current_prediction_features = self.memory_mechanism.build_prediction_features(
                    memory_state=memory_state,
                    probability_floor=self.configuration.probability_floor,
                )
                evaluation_context = DomainSpecificLanguageEvaluationContext(
                    character_vocabulary=character_vocabulary,
                    prediction_features=current_prediction_features,
                    probability_floor=self.configuration.probability_floor,
                )

                # ---- Evaluate all active predictors in parallel ----
                future_by_entry: dict[Any, ActivePoolEntry] = {}
                for entry in active_pool:
                    future = executor.submit(
                        entry.predictor_program.predict_character_distribution,
                        evaluation_context,
                        self.primitive_library,
                        frozen_store,
                    )
                    future_by_entry[future] = entry

                distributions_by_instance_identifier: dict[str, numpy.ndarray] = {}
                for future in as_completed(list(future_by_entry.keys())):
                    entry = future_by_entry[future]
                    distribution = future.result()
                    distributions_by_instance_identifier[entry.instance_identifier] = distribution

                # ---- Compute mixture prediction ----
                logarithmic_weights = [entry.logarithmic_weight_base_two for entry in active_pool]
                normalized_logarithmic_weights = _normalize_logarithmic_weights_base_two(logarithmic_weights)
                linear_weights = [2.0 ** value for value in normalized_logarithmic_weights]

                mixture = numpy.zeros(character_vocabulary.size, dtype=numpy.float64)
                for weight, entry in zip(linear_weights, active_pool):
                    mixture += float(weight) * distributions_by_instance_identifier[entry.instance_identifier]

                mixture_total = float(mixture.sum())
                if mixture_total <= 0.0:
                    mixture = numpy.ones(character_vocabulary.size, dtype=numpy.float64) / float(character_vocabulary.size)
                else:
                    mixture = mixture / mixture_total

                # Apply mixture-level probability floor.
                floor_value = float(self.configuration.probability_floor)
                if floor_value > 0.0:
                    uniform = numpy.ones(character_vocabulary.size, dtype=numpy.float64) / float(character_vocabulary.size)
                    mixture = (1.0 - floor_value) * mixture + floor_value * uniform
                    mixture = mixture / float(mixture.sum())

                predicted_probability = float(mixture[int(observed_character_index)])
                algorithm_loss_bits = -math.log2(predicted_probability)

                per_step_algorithm_loss_bits.append(algorithm_loss_bits)
                per_step_baseline_loss_bits.append(baseline_loss_per_character)

                # ---- Update weights using each predictor's probability on the observed character ----
                for entry in active_pool:
                    distribution = distributions_by_instance_identifier[entry.instance_identifier]
                    per_program_probability = float(distribution[int(observed_character_index)])
                    entry.logarithmic_weight_base_two += math.log2(per_program_probability)

                # ---- Update memory with the observation ----
                self.memory_mechanism.update(memory_state=memory_state, observed_character_index=int(observed_character_index))

                # ---- Update validation window ----
                validation_window.append(
                    ValidationExample(
                        prediction_features=current_prediction_features,
                        observed_character_index=int(observed_character_index),
                    )
                )
                if len(validation_window) > int(self.configuration.validation_window_length):
                    validation_window = validation_window[-int(self.configuration.validation_window_length) :]

                # ---- Determine incumbent (highest weight) ----
                incumbent_entry = max(active_pool, key=lambda entry: entry.logarithmic_weight_base_two)
                new_incumbent_signature = incumbent_entry.predictor_signature
                if new_incumbent_signature == incumbent_signature:
                    incumbent_run_length += 1
                else:
                    incumbent_signature = new_incumbent_signature
                    incumbent_run_length = 1

                # ---- Explore: run transformers to propose candidates ----
                sampled_transformers = self._sample_transformers_for_exploration()
                transformer_futures = [
                    executor.submit(
                        transformer.generate_candidate,
                        memory_state,
                        frozen_store,
                        self.primitive_library,
                        self.memory_mechanism,
                    )
                    for transformer in sampled_transformers
                ]
                for future in as_completed(transformer_futures):
                    candidate = future.result()
                    if candidate is None:
                        continue
                    # Do not re-add if already present.
                    if candidate.candidate_signature in candidate_buffer_by_signature:
                        continue
                    if any(entry.predictor_signature == candidate.candidate_signature for entry in active_pool):
                        continue
                    candidate_buffer_by_signature[candidate.candidate_signature] = candidate
                    # Keep buffer bounded.
                    if len(candidate_buffer_by_signature) > int(self.configuration.candidate_buffer_capacity):
                        candidate_buffer_by_signature.pop(next(iter(candidate_buffer_by_signature.keys())))

                # ---- Validate candidates on the recent window (parallel across candidates) ----
                if (
                    len(validation_window) > 4
                    and len(candidate_buffer_by_signature) > 0
                    and (int(time_step_index) % int(self.configuration.candidate_validation_interval) == 0)
                ):
                    baseline_loss_on_window = float(len(validation_window)) * baseline_loss_per_character

                    def compute_candidate_loss_bits(candidate: TransformerCandidate) -> float:
                        loss = 0.0
                        for example in validation_window:
                            context = DomainSpecificLanguageEvaluationContext(
                                character_vocabulary=character_vocabulary,
                                prediction_features=example.prediction_features,
                                probability_floor=self.configuration.probability_floor,
                            )
                            distribution = candidate.predictor_program.predict_character_distribution(
                                context,
                                self.primitive_library,
                                frozen_store,
                            )
                            probability_value = float(distribution[int(example.observed_character_index)])
                            loss += -math.log2(probability_value)
                        return loss

                    candidate_loss_futures = {}
                    for signature, candidate in candidate_buffer_by_signature.items():
                        candidate_loss_futures[executor.submit(compute_candidate_loss_bits, candidate)] = (signature, candidate)

                    accepted_candidates: list[tuple[str, TransformerCandidate, float]] = []
                    for future in as_completed(list(candidate_loss_futures.keys())):
                        signature, candidate = candidate_loss_futures[future]
                        loss_bits = float(future.result())
                        minimum_description_length_score = loss_bits + float(candidate.transformer_description_length_bits)
                        if minimum_description_length_score <= baseline_loss_on_window - float(self.configuration.detectability_slack_bits):
                            accepted_candidates.append((signature, candidate, minimum_description_length_score))

                    # Insert best candidates first (lowest minimum description length score).
                    accepted_candidates.sort(key=lambda item: item[2])

                    for signature, candidate, minimum_description_length_score in accepted_candidates:
                        # Insert with newborn weight.
                        logarithmic_total_weight_base_two = _logarithmic_sum_base_two(
                            entry.logarithmic_weight_base_two for entry in active_pool
                        )
                        initial_logarithmic_weight_base_two = self.newborn_weight_assignment_policy.compute_initial_log_weight_base_two(
                            log_total_weight_base_two=logarithmic_total_weight_base_two,
                            transformer_description_length_bits=candidate.transformer_description_length_bits,
                        )
                        new_entry = ActivePoolEntry(
                            predictor_program=candidate.predictor_program,
                            predictor_signature=candidate.candidate_signature,
                            origin_transformer_description_length_bits=candidate.transformer_description_length_bits,
                            logarithmic_weight_base_two=float(initial_logarithmic_weight_base_two),
                            instance_identifier=f"active_instance_{next_active_instance_identifier_index}",
                        )
                        next_active_instance_identifier_index += 1
                        active_pool.append(new_entry)
                        # Remove from buffer so we do not repeatedly insert the same candidate.
                        candidate_buffer_by_signature.pop(signature, None)

                        # Enforce pool capacity (evict lowest weight but protect incumbent).
                        while len(active_pool) > int(self.configuration.pool_capacity):
                            protected_incumbent_entry = max(active_pool, key=lambda entry: entry.logarithmic_weight_base_two)
                            removable_entries = [entry for entry in active_pool if entry is not protected_incumbent_entry]
                            if len(removable_entries) == 0:
                                break
                            entry_to_remove = min(removable_entries, key=lambda entry: entry.logarithmic_weight_base_two)
                            active_pool.remove(entry_to_remove)

                # ---- Freeze (optional) ----
                if (
                    len(validation_window) >= 8
                    and (int(time_step_index) % int(self.configuration.freeze_evaluation_interval) == 0)
                ):
                    incumbent_loss_bits = 0.0
                    for example in validation_window:
                        context = DomainSpecificLanguageEvaluationContext(
                            character_vocabulary=character_vocabulary,
                            prediction_features=example.prediction_features,
                            probability_floor=self.configuration.probability_floor,
                        )
                        distribution = incumbent_entry.predictor_program.predict_character_distribution(
                            context,
                            self.primitive_library,
                            frozen_store,
                        )
                        incumbent_loss_bits += -math.log2(float(distribution[int(example.observed_character_index)]))
                    incumbent_average_loss_bits = incumbent_loss_bits / float(len(validation_window))
                    baseline_average_loss_bits = baseline_loss_per_character

                    if self.freezing_policy.should_freeze(
                        time_step_index=int(time_step_index),
                        incumbent_run_length=int(incumbent_run_length),
                        incumbent_average_loss_bits=float(incumbent_average_loss_bits),
                        baseline_average_loss_bits=float(baseline_average_loss_bits),
                    ):
                        frozen_identifier = frozen_store.add_program(
                            predictor_program=incumbent_entry.predictor_program,
                            transformer_description_length_bits=incumbent_entry.origin_transformer_description_length_bits,
                        )
                        recall_key = self.memory_mechanism.build_recall_key(memory_state)
                        self.memory_mechanism.record_frozen_program_identifier(
                            memory_state=memory_state,
                            recall_key=recall_key,
                            frozen_program_identifier=frozen_identifier,
                        )

                per_step_incumbent_signature.append(incumbent_signature)
                active_pool_size_over_time.append(len(active_pool))
                frozen_store_size_over_time.append(frozen_store.size)

        return ResourceBoundedIncrementalInductionRunResult(
            character_vocabulary=character_vocabulary,
            per_step_algorithm_loss_bits=per_step_algorithm_loss_bits,
            per_step_baseline_loss_bits=per_step_baseline_loss_bits,
            per_step_incumbent_signature=per_step_incumbent_signature,
            active_pool_size_over_time=active_pool_size_over_time,
            frozen_store_size_over_time=frozen_store_size_over_time,
        )
