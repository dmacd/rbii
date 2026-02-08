from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as numpy

from character_vocabulary import CharacterVocabulary
from domain_specific_language import PredictionFeatures


class MemoryMechanism(Protocol):
    def initialize(self, character_vocabulary: CharacterVocabulary) -> "MemoryStateProtocol": ...

    def update(self, memory_state: "MemoryStateProtocol", observed_character_index: int) -> None: ...

    def build_prediction_features(
        self,
        memory_state: "MemoryStateProtocol",
        probability_floor: float,
    ) -> PredictionFeatures: ...

    def build_recall_key(self, memory_state: "MemoryStateProtocol") -> tuple[int, ...]: ...

    def record_frozen_program_identifier(
        self,
        memory_state: "MemoryStateProtocol",
        recall_key: tuple[int, ...],
        frozen_program_identifier: str,
    ) -> None: ...

    def recall_program_identifiers(
        self,
        memory_state: "MemoryStateProtocol",
        recall_key: tuple[int, ...],
        maximum_number: int,
    ) -> list[str]: ...


class MemoryStateProtocol(Protocol):
    character_vocabulary: CharacterVocabulary


@dataclass
class CharacterHistoryMemoryState:
    character_vocabulary: CharacterVocabulary
    maximum_context_length: int
    recall_key_length: int
    smoothing_alpha: float

    recent_character_indices: list[int]
    bigram_counts: numpy.ndarray
    trigram_counts: dict[tuple[int, int], numpy.ndarray]
    recall_index: dict[tuple[int, ...], list[str]]

    time_step_index: int = 0


class CharacterHistoryMemoryMechanism:
    def __init__(
        self,
        maximum_context_length: int = 16,
        recall_key_length: int = 4,
        smoothing_alpha: float = 0.5,
        maximum_program_identifiers_per_key: int = 8,
    ) -> None:
        if recall_key_length <= 0:
            raise ValueError("recall_key_length must be positive.")
        if maximum_context_length < recall_key_length:
            raise ValueError("maximum_context_length must be >= recall_key_length.")

        self.maximum_context_length = int(maximum_context_length)
        self.recall_key_length = int(recall_key_length)
        self.smoothing_alpha = float(smoothing_alpha)
        self.maximum_program_identifiers_per_key = int(maximum_program_identifiers_per_key)

    def initialize(self, character_vocabulary: CharacterVocabulary) -> CharacterHistoryMemoryState:
        vocabulary_size = character_vocabulary.size
        bigram_counts = numpy.zeros((vocabulary_size, vocabulary_size), dtype=numpy.int64)
        trigram_counts: dict[tuple[int, int], numpy.ndarray] = {}
        recall_index: dict[tuple[int, ...], list[str]] = {}
        return CharacterHistoryMemoryState(
            character_vocabulary=character_vocabulary,
            maximum_context_length=self.maximum_context_length,
            recall_key_length=self.recall_key_length,
            smoothing_alpha=self.smoothing_alpha,
            recent_character_indices=[],
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            recall_index=recall_index,
        )

    def update(self, memory_state: CharacterHistoryMemoryState, observed_character_index: int) -> None:
        # Update bigram and trigram counts based on the most recent context.
        if len(memory_state.recent_character_indices) >= 1:
            previous_character_index = memory_state.recent_character_indices[-1]
            memory_state.bigram_counts[previous_character_index, observed_character_index] += 1

        if len(memory_state.recent_character_indices) >= 2:
            previous_two_character_index = memory_state.recent_character_indices[-2]
            previous_character_index = memory_state.recent_character_indices[-1]
            key = (previous_two_character_index, previous_character_index)
            if key not in memory_state.trigram_counts:
                memory_state.trigram_counts[key] = numpy.zeros(memory_state.character_vocabulary.size, dtype=numpy.int64)
            memory_state.trigram_counts[key][observed_character_index] += 1

        memory_state.recent_character_indices.append(int(observed_character_index))
        if len(memory_state.recent_character_indices) > memory_state.maximum_context_length:
            memory_state.recent_character_indices = memory_state.recent_character_indices[-memory_state.maximum_context_length :]

        memory_state.time_step_index += 1

    def _counts_to_probability_distribution(
        self,
        counts: numpy.ndarray,
        smoothing_alpha: float,
    ) -> numpy.ndarray:
        smoothed = counts.astype(numpy.float64) + float(smoothing_alpha)
        total = float(smoothed.sum())
        if total <= 0.0:
            return numpy.ones_like(smoothed) / float(smoothed.shape[0])
        return smoothed / total

    def build_prediction_features(
        self,
        memory_state: CharacterHistoryMemoryState,
        probability_floor: float,
    ) -> PredictionFeatures:
        vocabulary_size = memory_state.character_vocabulary.size
        uniform_distribution = numpy.ones(vocabulary_size, dtype=numpy.float64) / float(vocabulary_size)

        if len(memory_state.recent_character_indices) >= 1:
            previous_character_index = memory_state.recent_character_indices[-1]
            bigram_counts = memory_state.bigram_counts[previous_character_index]
            bigram_distribution = self._counts_to_probability_distribution(bigram_counts, memory_state.smoothing_alpha)
        else:
            bigram_distribution = uniform_distribution

        if len(memory_state.recent_character_indices) >= 2:
            previous_two_character_index = memory_state.recent_character_indices[-2]
            previous_character_index = memory_state.recent_character_indices[-1]
            key = (previous_two_character_index, previous_character_index)
            if key in memory_state.trigram_counts:
                trigram_counts = memory_state.trigram_counts[key]
                trigram_distribution = self._counts_to_probability_distribution(trigram_counts, memory_state.smoothing_alpha)
            else:
                trigram_distribution = bigram_distribution
        else:
            trigram_distribution = bigram_distribution

        # Apply probability floor (consistent with the mixture floor).
        floor_value = float(max(0.0, probability_floor))
        if floor_value > 0.0:
            bigram_distribution = (1.0 - floor_value) * bigram_distribution + floor_value * uniform_distribution
            trigram_distribution = (1.0 - floor_value) * trigram_distribution + floor_value * uniform_distribution

        bigram_distribution = bigram_distribution / float(bigram_distribution.sum())
        trigram_distribution = trigram_distribution / float(trigram_distribution.sum())

        return PredictionFeatures(
            recent_character_indices=tuple(memory_state.recent_character_indices),
            bigram_probability_distribution=bigram_distribution,
            trigram_probability_distribution=trigram_distribution,
        )

    def build_recall_key(self, memory_state: CharacterHistoryMemoryState) -> tuple[int, ...]:
        # Key is the most recent `recall_key_length` indices.
        recent = memory_state.recent_character_indices
        if len(recent) >= memory_state.recall_key_length:
            return tuple(recent[-memory_state.recall_key_length :])
        # Pad with -1 to keep the key length fixed.
        padding = (-1,) * (memory_state.recall_key_length - len(recent))
        return tuple(padding + tuple(recent))

    def record_frozen_program_identifier(
        self,
        memory_state: CharacterHistoryMemoryState,
        recall_key: tuple[int, ...],
        frozen_program_identifier: str,
    ) -> None:
        existing = memory_state.recall_index.get(recall_key, [])
        # Move-to-front
        updated = [frozen_program_identifier] + [identifier for identifier in existing if identifier != frozen_program_identifier]
        memory_state.recall_index[recall_key] = updated[: self.maximum_program_identifiers_per_key]

    def recall_program_identifiers(
        self,
        memory_state: CharacterHistoryMemoryState,
        recall_key: tuple[int, ...],
        maximum_number: int,
    ) -> list[str]:
        existing = memory_state.recall_index.get(recall_key, [])
        return existing[: int(maximum_number)]
