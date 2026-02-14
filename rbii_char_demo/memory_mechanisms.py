from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

try:  # pragma: no cover
  from character_vocabulary import CharacterVocabulary
except ImportError:  # pragma: no cover
  from character_vocabulary import character_vocabulary as CharacterVocabulary

from domain_specific_language import PredictionFeatures


class MemoryStateProtocol(Protocol):
  pass


class MemoryMechanism(Protocol):
  def initialize(self,
                 character_vocabulary: CharacterVocabulary) -> MemoryStateProtocol: ...

  def update(self, memory_state: MemoryStateProtocol,
             observed_character_index: int) -> None: ...

  def build_prediction_features(self, memory_state: MemoryStateProtocol,
                                probability_floor: float) -> PredictionFeatures: ...

  def build_recall_key(self, memory_state: MemoryStateProtocol) -> str: ...

  def record_frozen_program_identifier(
      self, memory_state: MemoryStateProtocol, recall_key: str,
      frozen_program_identifier: str
  ) -> None: ...

  def recall_program_identifiers(
      self,
      memory_state: MemoryStateProtocol,
      recall_key: str,
      maximum_number_of_program_identifiers: int,
  ) -> list[str]: ...


@dataclass
class CharacterHistoryMemoryState:
  character_vocabulary: CharacterVocabulary
  maximum_context_length: int
  maximum_history_length: int
  recall_key_length: int
  maximum_program_identifiers_per_key: int

  recent_character_indices: list[int]
  history_character_indices: list[int]
  recall_index: dict[str, list[str]]
  time_step_index: int = 0


class CharacterHistoryMemoryMechanism:
  """Stores raw character history. No explicit n-gram statistics."""

  def __init__(
      self,
      maximum_context_length: int = 16,
      maximum_history_length: int = 2048,
      recall_key_length: int = 4,
      maximum_program_identifiers_per_key: int = 8,
  ) -> None:
    self.maximum_context_length = int(maximum_context_length)
    self.maximum_history_length = int(maximum_history_length)
    self.recall_key_length = int(recall_key_length)
    self.maximum_program_identifiers_per_key = int(
      maximum_program_identifiers_per_key)

  def initialize(self,
                 character_vocabulary: CharacterVocabulary) -> CharacterHistoryMemoryState:
    return CharacterHistoryMemoryState(
      character_vocabulary=character_vocabulary,
      maximum_context_length=self.maximum_context_length,
      maximum_history_length=self.maximum_history_length,
      recall_key_length=self.recall_key_length,
      maximum_program_identifiers_per_key=self.maximum_program_identifiers_per_key,
      recent_character_indices=[],
      history_character_indices=[],
      recall_index={},
      time_step_index=0,
    )

  def update(self, memory_state: CharacterHistoryMemoryState,
             observed_character_index: int) -> None:
    index = int(observed_character_index)
    memory_state.history_character_indices.append(index)
    if len(
        memory_state.history_character_indices) > memory_state.maximum_history_length:
      memory_state.history_character_indices = memory_state.history_character_indices[
        -memory_state.maximum_history_length:]

    memory_state.recent_character_indices.append(index)
    if len(
        memory_state.recent_character_indices) > memory_state.maximum_context_length:
      memory_state.recent_character_indices = memory_state.recent_character_indices[
        -memory_state.maximum_context_length:]

    memory_state.time_step_index += 1

  def build_prediction_features(self, memory_state: CharacterHistoryMemoryState,
                                probability_floor: float) -> PredictionFeatures:
    return PredictionFeatures(
      recent_character_indices=tuple(
        int(value) for value in memory_state.recent_character_indices),
      history_character_indices=tuple(
        int(value) for value in memory_state.history_character_indices),
      time_step_index=int(memory_state.time_step_index),
    )

  def build_recall_key(self, memory_state: CharacterHistoryMemoryState) -> str:
    if memory_state.recall_key_length <= 0:
      return ""
    key_indices = memory_state.recent_character_indices[
      -memory_state.recall_key_length:]
    return "_".join(str(int(value)) for value in key_indices)

  def record_frozen_program_identifier(
      self,
      memory_state: CharacterHistoryMemoryState,
      recall_key: str,
      frozen_program_identifier: str,
  ) -> None:
    key = str(recall_key)
    identifier = str(frozen_program_identifier)
    if key == "" or identifier == "":
      return
    memory_state.recall_index.setdefault(key, []).append(identifier)
    memory_state.recall_index[key] = memory_state.recall_index[key][
      -memory_state.maximum_program_identifiers_per_key:]

  def recall_program_identifiers(
      self,
      memory_state: CharacterHistoryMemoryState,
      recall_key: str,
      maximum_number_of_program_identifiers: int,
  ) -> list[str]:
    key = str(recall_key)
    identifiers = list(memory_state.recall_index.get(key, []))
    maximum_number = max(0, int(maximum_number_of_program_identifiers))
    if maximum_number == 0:
      return []
    return identifiers[-maximum_number:][::-1]


def character_history_memory_mechanism(
    maximum_context_length: int = 16,
    maximum_history_length: int = 2048,
    recall_key_length: int = 4,
    maximum_program_identifiers_per_key: int = 8,
) -> CharacterHistoryMemoryMechanism:
  """Compatibility factory."""
  return CharacterHistoryMemoryMechanism(
    maximum_context_length=maximum_context_length,
    maximum_history_length=maximum_history_length,
    recall_key_length=recall_key_length,
    maximum_program_identifiers_per_key=maximum_program_identifiers_per_key,
  )
