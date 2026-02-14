from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy

from domain_specific_language import (
  DomainSpecificLanguageEvaluationContext,
  DomainSpecificLanguageExpression,
  FrozenProgramStoreProtocol,
  is_probability_distribution,
)


class PredictorProgram(Protocol):
  def predict_character_distribution(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: Any,
      frozen_store: FrozenProgramStoreProtocol,
  ) -> numpy.ndarray: ...


@dataclass(frozen=True)
class DomainSpecificLanguagePredictorProgram:
  expression: DomainSpecificLanguageExpression

  def predict_character_distribution(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: Any,
      frozen_store: FrozenProgramStoreProtocol,
  ) -> numpy.ndarray:
    vocabulary_size = int(evaluation_context.character_vocabulary.size)

    try:
      value = self.expression.evaluate(
        evaluation_context=evaluation_context,
        primitive_library=primitive_library,
        frozen_store=frozen_store,
        environment=(),
      )
      distribution = numpy.asarray(value, dtype=numpy.float64)
    except Exception:
      distribution = numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)

    if not is_probability_distribution(distribution) or distribution.shape[
      0] != vocabulary_size:
      distribution = numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)

    probability_floor = float(evaluation_context.probability_floor)
    if probability_floor > 0.0:
      distribution = numpy.maximum(distribution, probability_floor)
      total = float(distribution.sum())
      if total > 0.0:
        distribution = distribution / total

    return distribution


def domain_specific_language_predictor_program(
    expression: DomainSpecificLanguageExpression) -> DomainSpecificLanguagePredictorProgram:
  return DomainSpecificLanguagePredictorProgram(expression=expression)
