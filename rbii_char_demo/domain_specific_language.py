from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as numpy

from character_vocabulary import CharacterVocabulary
from primitive_library import PrimitiveLibrary


@dataclass(frozen=True)
class PredictionFeatures:
  recent_character_indices: tuple[int, ...]
  bigram_probability_distribution: numpy.ndarray
  trigram_probability_distribution: numpy.ndarray


@dataclass(frozen=True)
class DomainSpecificLanguageEvaluationContext:
  character_vocabulary: CharacterVocabulary
  prediction_features: PredictionFeatures
  probability_floor: float


class FrozenProgramStore(Protocol):
  def get_program(self, program_identifier: str) -> Any: ...


class DomainSpecificLanguageExpression(Protocol):
  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibrary,
      frozen_store: FrozenProgramStore,
  ) -> Any: ...


@dataclass(frozen=True)
class ConstantExpression:
  value: Any

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibrary,
      frozen_store: FrozenProgramStore,
  ) -> Any:
    return self.value


@dataclass(frozen=True)
class PrimitiveCallExpression:
  primitive_name: str
  argument_expressions: tuple[DomainSpecificLanguageExpression, ...]

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibrary,
      frozen_store: FrozenProgramStore,
  ) -> Any:
    definition = primitive_library.get_definition(self.primitive_name)
    evaluated_arguments = [
      argument_expression.evaluate(evaluation_context, primitive_library,
                                   frozen_store)
      for argument_expression in self.argument_expressions
    ]
    return definition.callable_function(evaluation_context,
                                        *evaluated_arguments)


@dataclass(frozen=True)
class FrozenProgramCallExpression:
  frozen_program_identifier: str

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibrary,
      frozen_store: FrozenProgramStore,
  ) -> Any:
    program = frozen_store.get_program(self.frozen_program_identifier)
    distribution = program.predict_character_distribution(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
    )
    return distribution
