from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy

from domain_specific_language import (
  BOOLEAN_TYPE,
  DISTRIBUTION_TYPE,
  INTEGER_LIST_TYPE,
  INTEGER_PAIR_LIST_TYPE,
  INTEGER_PAIR_TYPE,
  INTEGER_TYPE,
  REAL_TYPE,
  STRING_TYPE,
  FunctionType,
  TypeSignatureValue,
  apply_callable_value,
)


@dataclass(frozen=True)
class PrimitiveDefinition:
  callable_function: Callable[..., Any]
  argument_type_signatures: tuple[TypeSignatureValue, ...]
  return_type_signature: TypeSignatureValue
  log_probability_base_two: float

  @property
  def description_length_bits(self) -> float:
    return -float(self.log_probability_base_two)


class PrimitiveLibrary:
  def __init__(self) -> None:
    self._definitions: dict[str, PrimitiveDefinition] = {}

  def register_primitive(
      self,
      name: str,
      callable_function: Callable[..., Any],
      argument_type_signatures: tuple[TypeSignatureValue, ...],
      return_type_signature: TypeSignatureValue,
      log_probability_base_two: float,
  ) -> None:
    self._definitions[name] = PrimitiveDefinition(
      callable_function=callable_function,
      argument_type_signatures=argument_type_signatures,
      return_type_signature=return_type_signature,
      log_probability_base_two=float(log_probability_base_two),
    )

  def get_definition(self, name: str) -> PrimitiveDefinition:
    if name not in self._definitions:
      raise KeyError(f"Unknown primitive: {name}")
    return self._definitions[name]

  def list_primitive_names(self) -> list[str]:
    return sorted(self._definitions.keys())

  def list_definitions(self) -> list[PrimitiveDefinition]:
    return [self._definitions[name] for name in self.list_primitive_names()]


def create_default_primitive_library() -> PrimitiveLibrary:
  library = PrimitiveLibrary()

  # -----------------------------
  # Core arithmetic and logic
  # -----------------------------

  def equals_integers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: int,
      right_value: int,
  ) -> bool:
    return int(left_value) == int(right_value)

  def less_than_integers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: int,
      right_value: int,
  ) -> bool:
    return int(left_value) < int(right_value)

  def add_integers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: int,
      right_value: int,
  ) -> int:
    return int(left_value) + int(right_value)

  def subtract_integers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: int,
      right_value: int,
  ) -> int:
    return int(left_value) - int(right_value)

  def modulo_integers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: int,
      right_value: int,
  ) -> int:
    right_value_integer = int(right_value)
    if right_value_integer == 0:
      return 0
    return int(left_value) % right_value_integer

  def add_real_numbers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: float,
      right_value: float,
  ) -> float:
    return float(left_value) + float(right_value)

  def subtract_real_numbers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: float,
      right_value: float,
  ) -> float:
    return float(left_value) - float(right_value)

  def multiply_real_numbers(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      left_value: float,
      right_value: float,
  ) -> float:
    return float(left_value) * float(right_value)

  def clamp_real_number_to_unit_interval(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      value: float,
  ) -> float:
    return float(max(0.0, min(1.0, float(value))))

  library.register_primitive(
    name="equals_integers",
    callable_function=equals_integers,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_TYPE),
    return_type_signature=BOOLEAN_TYPE,
    log_probability_base_two=-2.0,
  )
  library.register_primitive(
    name="less_than_integers",
    callable_function=less_than_integers,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_TYPE),
    return_type_signature=BOOLEAN_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="add_integers",
    callable_function=add_integers,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_TYPE),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="subtract_integers",
    callable_function=subtract_integers,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_TYPE),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="modulo_integers",
    callable_function=modulo_integers,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_TYPE),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="add_real_numbers",
    callable_function=add_real_numbers,
    argument_type_signatures=(REAL_TYPE, REAL_TYPE),
    return_type_signature=REAL_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="subtract_real_numbers",
    callable_function=subtract_real_numbers,
    argument_type_signatures=(REAL_TYPE, REAL_TYPE),
    return_type_signature=REAL_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="multiply_real_numbers",
    callable_function=multiply_real_numbers,
    argument_type_signatures=(REAL_TYPE, REAL_TYPE),
    return_type_signature=REAL_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="clamp_real_number_to_unit_interval",
    callable_function=clamp_real_number_to_unit_interval,
    argument_type_signatures=(REAL_TYPE,),
    return_type_signature=REAL_TYPE,
    log_probability_base_two=-4.0,
  )

  # -----------------------------
  # List and pair primitives
  # -----------------------------

  def adjacent_integer_pairs(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_list: tuple[int, ...],
  ) -> tuple[tuple[int, int], ...]:
    if len(integer_list) < 2:
      return ()
    return tuple((int(left), int(right)) for left, right in
                 zip(integer_list[:-1], integer_list[1:]))

  def get_first_element_of_integer_pair(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_pair: tuple[int, int],
  ) -> int:
    return int(integer_pair[0])

  def get_second_element_of_integer_pair(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_pair: tuple[int, int],
  ) -> int:
    return int(integer_pair[1])

  def get_last_element_of_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_list: tuple[int, ...],
      default_value: int,
  ) -> int:
    if len(integer_list) == 0:
      return int(default_value)
    return int(integer_list[-1])

  def take_first_elements_of_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      number_of_elements: int,
      integer_list: tuple[int, ...],
  ) -> tuple[int, ...]:
    n = max(0, int(number_of_elements))
    return tuple(int(value) for value in integer_list[:n])

  def drop_first_elements_of_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      number_of_elements: int,
      integer_list: tuple[int, ...],
  ) -> tuple[int, ...]:
    n = max(0, int(number_of_elements))
    return tuple(int(value) for value in integer_list[n:])

  def get_length_of_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_list: tuple[int, ...],
  ) -> int:
    return int(len(integer_list))

  def filter_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      predicate_function: Any,
      integer_list: tuple[int, ...],
  ) -> tuple[int, ...]:
    kept: list[int] = []
    for value in integer_list:
      predicate_value = apply_callable_value(
        callable_value=predicate_function,
        evaluation_context=evaluation_context,
        argument_value=int(value),
      )
      if bool(predicate_value):
        kept.append(int(value))
    return tuple(kept)

  def filter_integer_pair_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      predicate_function: Any,
      integer_pair_list: tuple[tuple[int, int], ...],
  ) -> tuple[tuple[int, int], ...]:
    kept: list[tuple[int, int]] = []
    for integer_pair in integer_pair_list:
      predicate_value = apply_callable_value(
        callable_value=predicate_function,
        evaluation_context=evaluation_context,
        argument_value=integer_pair,
      )
      if bool(predicate_value):
        kept.append((int(integer_pair[0]), int(integer_pair[1])))
    return tuple(kept)

  def map_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      mapping_function: Any,
      integer_list: tuple[int, ...],
  ) -> tuple[int, ...]:
    mapped: list[int] = []
    for value in integer_list:
      mapped_value = apply_callable_value(
        callable_value=mapping_function,
        evaluation_context=evaluation_context,
        argument_value=int(value),
      )
      mapped.append(int(mapped_value))
    return tuple(mapped)

  def map_integer_pair_list_to_integer_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      mapping_function: Any,
      integer_pair_list: tuple[tuple[int, int], ...],
  ) -> tuple[int, ...]:
    mapped: list[int] = []
    for integer_pair in integer_pair_list:
      mapped_value = apply_callable_value(
        callable_value=mapping_function,
        evaluation_context=evaluation_context,
        argument_value=integer_pair,
      )
      mapped.append(int(mapped_value))
    return tuple(mapped)

  def second_elements_of_integer_pair_list(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      integer_pair_list: tuple[tuple[int, int], ...],
  ) -> tuple[int, ...]:
    return tuple(int(integer_pair[1]) for integer_pair in integer_pair_list)

  library.register_primitive(
    name="adjacent_integer_pairs",
    callable_function=adjacent_integer_pairs,
    argument_type_signatures=(INTEGER_LIST_TYPE,),
    return_type_signature=INTEGER_PAIR_LIST_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="get_first_element_of_integer_pair",
    callable_function=get_first_element_of_integer_pair,
    argument_type_signatures=(INTEGER_PAIR_TYPE,),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="get_second_element_of_integer_pair",
    callable_function=get_second_element_of_integer_pair,
    argument_type_signatures=(INTEGER_PAIR_TYPE,),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="get_last_element_of_integer_list",
    callable_function=get_last_element_of_integer_list,
    argument_type_signatures=(INTEGER_LIST_TYPE, INTEGER_TYPE),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="take_first_elements_of_integer_list",
    callable_function=take_first_elements_of_integer_list,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_LIST_TYPE),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="drop_first_elements_of_integer_list",
    callable_function=drop_first_elements_of_integer_list,
    argument_type_signatures=(INTEGER_TYPE, INTEGER_LIST_TYPE),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="get_length_of_integer_list",
    callable_function=get_length_of_integer_list,
    argument_type_signatures=(INTEGER_LIST_TYPE,),
    return_type_signature=INTEGER_TYPE,
    log_probability_base_two=-4.0,
  )
  library.register_primitive(
    name="filter_integer_list",
    callable_function=filter_integer_list,
    argument_type_signatures=(FunctionType(INTEGER_TYPE, BOOLEAN_TYPE),
                              INTEGER_LIST_TYPE),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-6.0,
  )
  library.register_primitive(
    name="filter_integer_pair_list",
    callable_function=filter_integer_pair_list,
    argument_type_signatures=(FunctionType(INTEGER_PAIR_TYPE, BOOLEAN_TYPE),
                              INTEGER_PAIR_LIST_TYPE),
    return_type_signature=INTEGER_PAIR_LIST_TYPE,
    log_probability_base_two=-6.0,
  )
  library.register_primitive(
    name="map_integer_list",
    callable_function=map_integer_list,
    argument_type_signatures=(FunctionType(INTEGER_TYPE, INTEGER_TYPE),
                              INTEGER_LIST_TYPE),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-6.0,
  )
  library.register_primitive(
    name="map_integer_pair_list_to_integer_list",
    callable_function=map_integer_pair_list_to_integer_list,
    argument_type_signatures=(FunctionType(INTEGER_PAIR_TYPE, INTEGER_TYPE),
                              INTEGER_PAIR_LIST_TYPE),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-6.0,
  )
  library.register_primitive(
    name="second_elements_of_integer_pair_list",
    callable_function=second_elements_of_integer_pair_list,
    argument_type_signatures=(INTEGER_PAIR_LIST_TYPE,),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-5.0,
  )

  # -----------------------------
  # Character and memory access
  # -----------------------------

  def recent_character_indices(evaluation_context: Any, primitive_library: Any,
                               frozen_store: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in
                 evaluation_context.prediction_features.recent_character_indices)

  def history_character_indices(evaluation_context: Any, primitive_library: Any,
                                frozen_store: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in
                 evaluation_context.prediction_features.history_character_indices)

  def recalled_frozen_program_identifier(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      position: int,
  ) -> str:
    position_integer = int(position)
    identifiers = evaluation_context.recalled_frozen_program_identifiers
    if position_integer < 0 or position_integer >= len(identifiers):
      return ""
    return str(identifiers[position_integer])

  library.register_primitive(
    name="recent_character_indices",
    callable_function=recent_character_indices,
    argument_type_signatures=(),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-2.0,
  )
  library.register_primitive(
    name="history_character_indices",
    callable_function=history_character_indices,
    argument_type_signatures=(),
    return_type_signature=INTEGER_LIST_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="recalled_frozen_program_identifier",
    callable_function=recalled_frozen_program_identifier,
    argument_type_signatures=(INTEGER_TYPE,),
    return_type_signature=STRING_TYPE,
    log_probability_base_two=-3.0,
  )

  # -----------------------------
  # Distribution primitives
  # -----------------------------

  def uniform_character_distribution(evaluation_context: Any,
                                     primitive_library: Any,
                                     frozen_store: Any) -> numpy.ndarray:
    vocabulary_size = int(evaluation_context.character_vocabulary.size)
    return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
      vocabulary_size)

  def one_hot_character_distribution(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      character_index: int,
  ) -> numpy.ndarray:
    vocabulary_size = int(evaluation_context.character_vocabulary.size)
    index = int(character_index)
    if index < 0 or index >= vocabulary_size:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    distribution = numpy.zeros(vocabulary_size, dtype=numpy.float64)
    distribution[index] = 1.0
    return distribution

  def mixture_of_two_distributions(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      first_distribution: numpy.ndarray,
      second_distribution: numpy.ndarray,
      first_weight: float,
  ) -> numpy.ndarray:
    weight = float(max(0.0, min(1.0, float(first_weight))))
    first = numpy.asarray(first_distribution, dtype=numpy.float64)
    second = numpy.asarray(second_distribution, dtype=numpy.float64)
    if first.shape != second.shape:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    mixed = weight * first + (1.0 - weight) * second
    total = float(mixed.sum())
    if not numpy.isfinite(total) or total <= 0.0:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    return mixed / total

  def normalized_histogram_distribution(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      character_indices: tuple[int, ...],
      smoothing_alpha: float,
  ) -> numpy.ndarray:
    vocabulary_size = int(evaluation_context.character_vocabulary.size)
    counts = numpy.zeros(vocabulary_size, dtype=numpy.float64)
    for index in character_indices:
      index_integer = int(index)
      if 0 <= index_integer < vocabulary_size:
        counts[index_integer] += 1.0
    counts += float(max(0.0, float(smoothing_alpha)))
    total = float(counts.sum())
    if not numpy.isfinite(total) or total <= 0.0:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    return counts / total

  def predict_with_frozen_program_identifier(
      evaluation_context: Any,
      primitive_library: Any,
      frozen_store: Any,
      program_identifier: str,
  ) -> numpy.ndarray:
    if not isinstance(program_identifier, str) or program_identifier == "":
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    try:
      program = frozen_store.get_program(program_identifier)
    except Exception:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)
    try:
      distribution = program.predict_character_distribution(
        evaluation_context=evaluation_context,
        primitive_library=primitive_library,
        frozen_store=frozen_store,
      )
      return numpy.asarray(distribution, dtype=numpy.float64)
    except Exception:
      return uniform_character_distribution(evaluation_context,
                                            primitive_library, frozen_store)

  library.register_primitive(
    name="uniform_character_distribution",
    callable_function=uniform_character_distribution,
    argument_type_signatures=(),
    return_type_signature=DISTRIBUTION_TYPE,
    log_probability_base_two=-1.0,
  )
  library.register_primitive(
    name="one_hot_character_distribution",
    callable_function=one_hot_character_distribution,
    argument_type_signatures=(INTEGER_TYPE,),
    return_type_signature=DISTRIBUTION_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="mixture_of_two_distributions",
    callable_function=mixture_of_two_distributions,
    argument_type_signatures=(DISTRIBUTION_TYPE, DISTRIBUTION_TYPE, REAL_TYPE),
    return_type_signature=DISTRIBUTION_TYPE,
    log_probability_base_two=-3.0,
  )
  library.register_primitive(
    name="normalized_histogram_distribution",
    callable_function=normalized_histogram_distribution,
    argument_type_signatures=(INTEGER_LIST_TYPE, REAL_TYPE),
    return_type_signature=DISTRIBUTION_TYPE,
    log_probability_base_two=-5.0,
  )
  library.register_primitive(
    name="predict_with_frozen_program_identifier",
    callable_function=predict_with_frozen_program_identifier,
    argument_type_signatures=(STRING_TYPE,),
    return_type_signature=DISTRIBUTION_TYPE,
    log_probability_base_two=-4.0,
  )

  # NOTE:
  # We intentionally do NOT register bigram/digram/trigram primitives.
  # Programs can learn those features by processing raw history.

  return library
