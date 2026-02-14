from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy

try:  # pragma: no cover
  # Style-updated codebase
  from character_vocabulary import CharacterVocabulary
except ImportError:  # pragma: no cover
  # Backwards-compatible import (original demo)
  from character_vocabulary import character_vocabulary as CharacterVocabulary


# -----------------------------
# DreamCoder-style type metadata
# -----------------------------


class TypeSignature(Protocol):
  def __str__(self) -> str: ...


@dataclass(frozen=True)
class BaseType:
  name: str

  def __str__(self) -> str:
    return self.name


@dataclass(frozen=True)
class FunctionType:
  argument_type: "TypeSignatureValue"
  return_type: "TypeSignatureValue"

  def __str__(self) -> str:
    return f"({self.argument_type} -> {self.return_type})"


@dataclass(frozen=True)
class ListType:
  element_type: "TypeSignatureValue"

  def __str__(self) -> str:
    return f"list[{self.element_type}]"


@dataclass(frozen=True)
class PairType:
  first_type: "TypeSignatureValue"
  second_type: "TypeSignatureValue"

  def __str__(self) -> str:
    return f"pair[{self.first_type}, {self.second_type}]"


TypeSignatureValue = BaseType | FunctionType | ListType | PairType

INTEGER_TYPE = BaseType("integer")
REAL_TYPE = BaseType("real")
BOOLEAN_TYPE = BaseType("boolean")
STRING_TYPE = BaseType("string")
DISTRIBUTION_TYPE = BaseType("distribution")

INTEGER_LIST_TYPE = ListType(INTEGER_TYPE)
INTEGER_PAIR_TYPE = PairType(INTEGER_TYPE, INTEGER_TYPE)
INTEGER_PAIR_LIST_TYPE = ListType(INTEGER_PAIR_TYPE)


# -----------------------------
# Evaluation context
# -----------------------------


@dataclass(frozen=True)
class PredictionFeatures:
  """Features exposed to programs.

  Notes:
      - `recent_character_indices` should be short (bounded context window).
      - `history_character_indices` can be larger and is intended to let programs
        *learn* n-gram-like structure by processing raw observations rather than
        relying on special n-gram primitives.
  """

  recent_character_indices: tuple[int, ...]
  history_character_indices: tuple[int, ...]
  time_step_index: int


@dataclass(frozen=True)
class DomainSpecificLanguageEvaluationContext:
  character_vocabulary: CharacterVocabulary
  prediction_features: PredictionFeatures
  recalled_frozen_program_identifiers: tuple[str, ...]
  probability_floor: float


# -----------------------------
# Protocols used by the evaluator
# -----------------------------


class FrozenProgramStoreProtocol(Protocol):
  def get_program(self, program_identifier: str) -> Any: ...


class PrimitiveLibraryProtocol(Protocol):
  def get_definition(self, name: str) -> Any: ...


class DomainSpecificLanguageExpression(Protocol):
  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any: ...


class CallableValueProtocol(Protocol):
  def apply(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      argument_value: Any,
  ) -> Any: ...


@dataclass(frozen=True)
class ClosureValue:
  """A first-class function value produced by `LambdaExpression`."""

  parameter_type_signature: TypeSignatureValue | None
  body_expression: DomainSpecificLanguageExpression
  captured_environment: tuple[Any, ...]
  primitive_library: PrimitiveLibraryProtocol
  frozen_store: FrozenProgramStoreProtocol

  def apply(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      argument_value: Any,
  ) -> Any:
    extended_environment = self.captured_environment + (argument_value,)
    return self.body_expression.evaluate(
      evaluation_context=evaluation_context,
      primitive_library=self.primitive_library,
      frozen_store=self.frozen_store,
      environment=extended_environment,
    )


def apply_callable_value(
    callable_value: Any,
    evaluation_context: DomainSpecificLanguageEvaluationContext,
    argument_value: Any,
) -> Any:
  """Applies either a DSL closure or a plain Python callable."""

  if hasattr(callable_value, "apply"):
    return callable_value.apply(evaluation_context, argument_value)

  if callable(callable_value):
    return callable_value(argument_value)

  raise TypeError("Value is not callable.")


# -----------------------------
# Expression nodes
# -----------------------------


@dataclass(frozen=True)
class ConstantExpression:
  value: Any

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    return self.value


@dataclass(frozen=True)
class VariableExpression:
  """De Bruijn-indexed variable.

  `variable_index=0` refers to the most recently bound variable.
  """

  variable_index: int

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    index_from_end = int(self.variable_index) + 1
    if index_from_end <= 0 or index_from_end > len(environment):
      raise IndexError("Variable index out of range.")
    return environment[-index_from_end]


@dataclass(frozen=True)
class LambdaExpression:
  parameter_type_signature: TypeSignatureValue | None
  body_expression: DomainSpecificLanguageExpression

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    return ClosureValue(
      parameter_type_signature=self.parameter_type_signature,
      body_expression=self.body_expression,
      captured_environment=environment,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
    )


@dataclass(frozen=True)
class ApplyExpression:
  function_expression: DomainSpecificLanguageExpression
  argument_expression: DomainSpecificLanguageExpression

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    function_value = self.function_expression.evaluate(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
      environment=environment,
    )
    argument_value = self.argument_expression.evaluate(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
      environment=environment,
    )
    return apply_callable_value(
      callable_value=function_value,
      evaluation_context=evaluation_context,
      argument_value=argument_value,
    )


@dataclass(frozen=True)
class IfExpression:
  condition_expression: DomainSpecificLanguageExpression
  then_expression: DomainSpecificLanguageExpression
  else_expression: DomainSpecificLanguageExpression

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    condition_value = self.condition_expression.evaluate(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
      environment=environment,
    )
    if bool(condition_value):
      return self.then_expression.evaluate(
        evaluation_context=evaluation_context,
        primitive_library=primitive_library,
        frozen_store=frozen_store,
        environment=environment,
      )
    return self.else_expression.evaluate(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
      environment=environment,
    )


@dataclass(frozen=True)
class PrimitiveCallExpression:
  primitive_name: str
  argument_expressions: tuple[DomainSpecificLanguageExpression, ...]

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    definition = primitive_library.get_definition(self.primitive_name)
    evaluated_arguments = [
      argument_expression.evaluate(
        evaluation_context=evaluation_context,
        primitive_library=primitive_library,
        frozen_store=frozen_store,
        environment=environment,
      )
      for argument_expression in self.argument_expressions
    ]
    return definition.callable_function(evaluation_context, primitive_library,
                                        frozen_store, *evaluated_arguments)


@dataclass(frozen=True)
class FrozenProgramCallExpression:
  frozen_program_identifier: str

  def evaluate(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      primitive_library: PrimitiveLibraryProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      environment: tuple[Any, ...] = (),
  ) -> Any:
    program = frozen_store.get_program(self.frozen_program_identifier)
    distribution = program.predict_character_distribution(
      evaluation_context=evaluation_context,
      primitive_library=primitive_library,
      frozen_store=frozen_store,
    )
    return distribution


def is_probability_distribution(distribution: Any) -> bool:
  if not isinstance(distribution, numpy.ndarray):
    return False
  if distribution.ndim != 1:
    return False
  total = float(distribution.sum())
  return numpy.isfinite(total) and total > 0.0
