from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as numpy

from domain_specific_language import (
    DomainSpecificLanguageEvaluationContext,
    DomainSpecificLanguageExpression,
)
from primitive_library import PrimitiveLibrary


class FrozenProgramStoreProtocol(Protocol):
    def get_program(self, program_identifier: str): ...


class PredictorProgramProtocol(Protocol):
    def predict_character_distribution(
        self,
        evaluation_context: DomainSpecificLanguageEvaluationContext,
        primitive_library: PrimitiveLibrary,
        frozen_store: FrozenProgramStoreProtocol,
    ) -> numpy.ndarray: ...


def _normalize_distribution(distribution: numpy.ndarray) -> numpy.ndarray:
    total = float(distribution.sum())
    if not numpy.isfinite(total) or total <= 0.0:
        raise ValueError("Distribution is not normalizable.")
    return distribution / total


def _apply_probability_floor(distribution: numpy.ndarray, probability_floor: float) -> numpy.ndarray:
    floor_value = float(probability_floor)
    if floor_value <= 0.0:
        return distribution
    vocabulary_size = distribution.shape[0]
    uniform = numpy.ones(vocabulary_size, dtype=numpy.float64) / float(vocabulary_size)
    floored = (1.0 - floor_value) * distribution + floor_value * uniform
    return _normalize_distribution(floored)


@dataclass(frozen=True)
class DomainSpecificLanguagePredictorProgram:
    expression: DomainSpecificLanguageExpression

    def predict_character_distribution(
        self,
        evaluation_context: DomainSpecificLanguageEvaluationContext,
        primitive_library: PrimitiveLibrary,
        frozen_store: FrozenProgramStoreProtocol,
    ) -> numpy.ndarray:
        result = self.expression.evaluate(evaluation_context, primitive_library, frozen_store)
        if not isinstance(result, numpy.ndarray):
            raise TypeError("Predictor expression must evaluate to a numpy.ndarray distribution.")
        normalized = _normalize_distribution(result.astype(numpy.float64))
        return _apply_probability_floor(normalized, evaluation_context.probability_floor)
