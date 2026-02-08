from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as numpy


@dataclass(frozen=True)
class PrimitiveDefinition:
  callable_function: Callable[..., Any]
  description_length_bits: int


class PrimitiveLibrary:
  def __init__(self) -> None:
    self._definitions: dict[str, PrimitiveDefinition] = {}

  def register_primitive(
      self,
      name: str,
      callable_function: Callable[..., Any],
      description_length_bits: int,
  ) -> None:
    self._definitions[name] = PrimitiveDefinition(
      callable_function=callable_function,
      description_length_bits=description_length_bits,
    )

  def get_definition(self, name: str) -> PrimitiveDefinition:
    if name not in self._definitions:
      raise KeyError(f"Unknown primitive: {name}")
    return self._definitions[name]

  def list_primitive_names(self) -> list[str]:
    return sorted(self._definitions.keys())


def create_default_primitive_library() -> PrimitiveLibrary:
  library = PrimitiveLibrary()

  # ---- Control and numeric primitives (minimal but useful) ----

  def if_then_else(condition: bool, then_value: Any, else_value: Any) -> Any:
    return then_value if condition else else_value

  def equals(left_value: Any, right_value: Any) -> bool:
    return left_value == right_value

  def add_real_numbers(left_value: float, right_value: float) -> float:
    return float(left_value) + float(right_value)

  def subtract_real_numbers(left_value: float, right_value: float) -> float:
    return float(left_value) - float(right_value)

  def multiply_real_numbers(left_value: float, right_value: float) -> float:
    return float(left_value) * float(right_value)

  library.register_primitive("if_then_else", if_then_else,
                             description_length_bits=1)
  library.register_primitive("equals", equals, description_length_bits=1)
  library.register_primitive("add_real_numbers", add_real_numbers,
                             description_length_bits=1)
  library.register_primitive("subtract_real_numbers", subtract_real_numbers,
                             description_length_bits=1)
  library.register_primitive("multiply_real_numbers", multiply_real_numbers,
                             description_length_bits=1)

  # ---- Character prediction primitives ----

  def uniform_character_distribution(evaluation_context: Any) -> numpy.ndarray:
    vocabulary_size = evaluation_context.character_vocabulary.size
    return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
      vocabulary_size)

  def bigram_character_distribution(evaluation_context: Any) -> numpy.ndarray:
    return evaluation_context.prediction_features.bigram_probability_distribution

  def trigram_character_distribution(evaluation_context: Any) -> numpy.ndarray:
    return evaluation_context.prediction_features.trigram_probability_distribution

  def mixture_of_two_character_distributions(
      evaluation_context: Any,
      first_distribution: numpy.ndarray,
      second_distribution: numpy.ndarray,
      first_weight: float,
  ) -> numpy.ndarray:
    first_weight_clamped = float(max(0.0, min(1.0, first_weight)))
    second_weight_clamped = 1.0 - first_weight_clamped
    return first_weight_clamped * first_distribution + second_weight_clamped * second_distribution

  def digit_cycle_character_distribution(evaluation_context: Any,
                                         step_size: int) -> numpy.ndarray:
    # If the most recent character is a digit, predict a modular cycle; else fall back to uniform.
    vocabulary = evaluation_context.character_vocabulary
    if len(evaluation_context.prediction_features.recent_character_indices) < 1:
      return uniform_character_distribution(evaluation_context)

    last_character_index = \
    evaluation_context.prediction_features.recent_character_indices[-1]
    last_character = vocabulary.index_to_character[last_character_index]
    if not last_character.isdigit():
      return uniform_character_distribution(evaluation_context)

    next_digit = (int(last_character) + int(step_size)) % 10
    next_character = str(next_digit)
    if next_character not in vocabulary.character_to_index:
      return uniform_character_distribution(evaluation_context)

    distribution = numpy.zeros(vocabulary.size, dtype=numpy.float64)
    distribution[vocabulary.character_to_index[next_character]] = 1.0
    return distribution

  library.register_primitive("uniform_character_distribution",
                             uniform_character_distribution,
                             description_length_bits=2)
  library.register_primitive("bigram_character_distribution",
                             bigram_character_distribution,
                             description_length_bits=3)
  library.register_primitive("trigram_character_distribution",
                             trigram_character_distribution,
                             description_length_bits=4)
  library.register_primitive("mixture_of_two_character_distributions",
                             mixture_of_two_character_distributions,
                             description_length_bits=3)
  library.register_primitive("digit_cycle_character_distribution",
                             digit_cycle_character_distribution,
                             description_length_bits=5)

  return library
