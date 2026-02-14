from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Protocol

from domain_specific_language import (
  ApplyExpression,
  BOOLEAN_TYPE,
  ConstantExpression,
  DISTRIBUTION_TYPE,
  DomainSpecificLanguageEvaluationContext,
  DomainSpecificLanguageExpression,
  FunctionType,
  IfExpression,
  INTEGER_LIST_TYPE,
  INTEGER_PAIR_LIST_TYPE,
  INTEGER_PAIR_TYPE,
  INTEGER_TYPE,
  LambdaExpression,
  PrimitiveCallExpression,
  REAL_TYPE,
  STRING_TYPE,
  TypeSignatureValue,
  VariableExpression,
)

from primitive_library import PrimitiveDefinition, PrimitiveLibrary


@dataclass(frozen=True)
class ConstantProductionCandidate:
  value: Any
  log_probability_base_two: float

  @property
  def description_length_bits(self) -> float:
    return -float(self.log_probability_base_two)


@dataclass(frozen=True)
class TransformerExpressionCandidate:
  """A transformer candidate in the sense of RBII.

  In this demo implementation, a "transformer" is an expression tree in the DSL
  whose evaluation yields a next-character probability distribution directly.
  This is consistent with the RBII paper's allowance that a transformer may
  "output a predictor program p (or directly output predictions; we treat this as outputting a p)".
  """

  transformer_expression: DomainSpecificLanguageExpression
  description_length_bits: float
  log_probability_base_two: float


class TransformerSearchStrategy(Protocol):
  def propose_transformer_expressions(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      maximum_number_of_expressions: int,
  ) -> list[TransformerExpressionCandidate]:
    """Returns up to `maximum_number_of_expressions` candidate transformer expressions.

    The returned candidates must include DreamCoder-style description length / log probability
    metadata, so RBII can:
        - validate via MDL-style window scoring
        - inject newborn weights consistent with a prior
    """


class ProbabilisticProgramGrammar:
  """DreamCoder-style probabilistic grammar over DSL expressions.

  Notes:
      - Primitive probabilities come from `PrimitiveDefinition.log_probability_base_two`.
      - Syntactic constructors (lambda/apply/if/variable/constant/primitive_call) have their own
        probabilities so you can later learn them (or condition them on context).
  """

  def __init__(
      self,
      primitive_library: PrimitiveLibrary,
      syntax_constructor_log_probability_base_two: dict[str, float],
      constant_candidates_by_type: dict[
        TypeSignatureValue, list[ConstantProductionCandidate]],
      application_argument_type_signatures: tuple[TypeSignatureValue, ...],
  ) -> None:
    self.primitive_library = primitive_library
    self.syntax_constructor_log_probability_base_two = dict(
      syntax_constructor_log_probability_base_two)
    self.constant_candidates_by_type = dict(constant_candidates_by_type)
    self.application_argument_type_signatures = tuple(
      application_argument_type_signatures)

    self._primitive_names_by_return_type: dict[
      TypeSignatureValue, list[str]] = {}
    for primitive_name in self.primitive_library.list_primitive_names():
      primitive_definition = self.primitive_library.get_definition(
        primitive_name)
      return_type_signature = primitive_definition.return_type_signature
      self._primitive_names_by_return_type.setdefault(return_type_signature,
                                                      []).append(primitive_name)

    for return_type_signature, primitive_names in self._primitive_names_by_return_type.items():
      primitive_names.sort(
        key=lambda name: self.primitive_library.get_definition(
          name).description_length_bits)

  def get_syntax_constructor_log_probability_base_two(self,
                                                      constructor_name: str) -> float:
    if constructor_name not in self.syntax_constructor_log_probability_base_two:
      raise KeyError(f"Unknown syntax constructor: {constructor_name}")
    return float(
      self.syntax_constructor_log_probability_base_two[constructor_name])

  def get_syntax_constructor_description_length_bits(self,
                                                     constructor_name: str) -> float:
    return -float(
      self.get_syntax_constructor_log_probability_base_two(constructor_name))

  def list_constant_candidates(self, type_signature: TypeSignatureValue) -> \
  list[ConstantProductionCandidate]:
    return list(self.constant_candidates_by_type.get(type_signature, []))

  def list_primitives_returning_type(self,
                                     type_signature: TypeSignatureValue) -> \
  list[tuple[str, PrimitiveDefinition]]:
    primitive_names = list(
      self._primitive_names_by_return_type.get(type_signature, []))
    return [(name, self.primitive_library.get_definition(name)) for name in
            primitive_names]

  def list_application_argument_types(self) -> tuple[TypeSignatureValue, ...]:
    return self.application_argument_type_signatures

  def list_known_type_signatures(self) -> list[TypeSignatureValue]:
    type_signatures: set[TypeSignatureValue] = set()

    for primitive_name in self.primitive_library.list_primitive_names():
      primitive_definition = self.primitive_library.get_definition(
        primitive_name)
      type_signatures.add(primitive_definition.return_type_signature)
      for argument_type_signature in primitive_definition.argument_type_signatures:
        type_signatures.add(argument_type_signature)

    for constant_type_signature in self.constant_candidates_by_type.keys():
      type_signatures.add(constant_type_signature)

    type_signatures.add(DISTRIBUTION_TYPE)
    type_signatures.add(INTEGER_TYPE)
    type_signatures.add(REAL_TYPE)
    type_signatures.add(BOOLEAN_TYPE)
    type_signatures.add(STRING_TYPE)
    type_signatures.add(INTEGER_LIST_TYPE)
    type_signatures.add(INTEGER_PAIR_TYPE)
    type_signatures.add(INTEGER_PAIR_LIST_TYPE)

    return sorted(type_signatures, key=str)


def create_default_probabilistic_program_grammar(
    primitive_library: PrimitiveLibrary) -> ProbabilisticProgramGrammar:
  # Syntactic constructor prior (DreamCoder-style); adjust or learn later.
  syntax_constructor_log_probability_base_two = {
    "variable": -2.0,
    "constant": -3.5,
    "lambda": -5.0,
    "apply": -5.0,
    "if": -6.0,
    "primitive_call": -1.5,
  }

  constant_candidates_by_type: dict[
    TypeSignatureValue, list[ConstantProductionCandidate]] = {
    INTEGER_TYPE: [
      ConstantProductionCandidate(value=0, log_probability_base_two=-2.0),
      ConstantProductionCandidate(value=1, log_probability_base_two=-2.0),
      ConstantProductionCandidate(value=2, log_probability_base_two=-3.0),
      ConstantProductionCandidate(value=3, log_probability_base_two=-3.5),
      ConstantProductionCandidate(value=4, log_probability_base_two=-4.0),
      ConstantProductionCandidate(value=5, log_probability_base_two=-4.0),
      ConstantProductionCandidate(value=8, log_probability_base_two=-5.0),
      ConstantProductionCandidate(value=16, log_probability_base_two=-6.0),
    ],
    REAL_TYPE: [
      ConstantProductionCandidate(value=0.0, log_probability_base_two=-2.5),
      ConstantProductionCandidate(value=0.1, log_probability_base_two=-4.0),
      ConstantProductionCandidate(value=0.25, log_probability_base_two=-4.5),
      ConstantProductionCandidate(value=0.5, log_probability_base_two=-3.5),
      ConstantProductionCandidate(value=1.0, log_probability_base_two=-3.5),
    ],
    BOOLEAN_TYPE: [
      ConstantProductionCandidate(value=True, log_probability_base_two=-2.0),
      ConstantProductionCandidate(value=False, log_probability_base_two=-2.0),
    ],
    STRING_TYPE: [
      ConstantProductionCandidate(value="", log_probability_base_two=-1.0),
    ],
  }

  # Limit application argument types to keep enumeration tractable.
  application_argument_type_signatures = (
    INTEGER_TYPE,
    REAL_TYPE,
    BOOLEAN_TYPE,
    STRING_TYPE,
    INTEGER_LIST_TYPE,
    INTEGER_PAIR_TYPE,
    INTEGER_PAIR_LIST_TYPE,
    DISTRIBUTION_TYPE,
  )

  return ProbabilisticProgramGrammar(
    primitive_library=primitive_library,
    syntax_constructor_log_probability_base_two=syntax_constructor_log_probability_base_two,
    constant_candidates_by_type=constant_candidates_by_type,
    application_argument_type_signatures=application_argument_type_signatures,
  )


def compute_expression_description_length_bits(
    grammar: ProbabilisticProgramGrammar,
    expression: DomainSpecificLanguageExpression,
) -> float:
  """Computes DreamCoder-style description length from the grammar.

  This is a structural code-length; for primitives it uses the primitive prior,
  and for constructors it uses the constructor prior.
  """

  if isinstance(expression, ConstantExpression):
    constant_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "constant")
    constant_candidates = grammar.list_constant_candidates(
      _infer_type_signature_from_value(expression.value))
    for candidate in constant_candidates:
      if candidate.value == expression.value:
        return constant_constructor_bits + candidate.description_length_bits
    return constant_constructor_bits + 10.0

  if isinstance(expression, VariableExpression):
    variable_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "variable")
    variable_index_penalty_bits = math.log2(
      float(expression.variable_index) + 2.0)
    return variable_constructor_bits + variable_index_penalty_bits

  if isinstance(expression, LambdaExpression):
    lambda_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "lambda")
    return lambda_constructor_bits + compute_expression_description_length_bits(
      grammar, expression.body_expression)

  if isinstance(expression, ApplyExpression):
    apply_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "apply")
    return apply_constructor_bits + compute_expression_description_length_bits(
      grammar, expression.function_expression
    ) + compute_expression_description_length_bits(grammar,
                                                   expression.argument_expression)

  if isinstance(expression, IfExpression):
    if_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "if")
    return if_constructor_bits + compute_expression_description_length_bits(
      grammar, expression.condition_expression
    ) + compute_expression_description_length_bits(grammar,
                                                   expression.then_expression) + compute_expression_description_length_bits(
      grammar, expression.else_expression
    )

  if isinstance(expression, PrimitiveCallExpression):
    primitive_call_constructor_bits = grammar.get_syntax_constructor_description_length_bits(
      "primitive_call")
    primitive_definition = grammar.primitive_library.get_definition(
      expression.primitive_name)
    primitive_bits = primitive_definition.description_length_bits
    argument_bits = sum(
      compute_expression_description_length_bits(grammar, argument_expression)
      for argument_expression in expression.argument_expressions
    )
    return primitive_call_constructor_bits + primitive_bits + argument_bits

  # Unknown node type: conservative penalty.
  return 100.0


def _infer_type_signature_from_value(value: Any) -> TypeSignatureValue:
  if isinstance(value, bool):
    return BOOLEAN_TYPE
  if isinstance(value, int):
    return INTEGER_TYPE
  if isinstance(value, float):
    return REAL_TYPE
  if isinstance(value, str):
    return STRING_TYPE
  return STRING_TYPE


class EnumerativeExpressionEnumerator:
  """Enumerates DSL expression trees within a probability budget.

  This is a top-down enumerator that recursively expands productions while budget remains.

  Important:
      - This is designed to be easy to swap out later for probabilistic sampling.
      - Enumeration order is biased toward lower description length bits.
  """

  def __init__(
      self,
      grammar: ProbabilisticProgramGrammar,
      target_type_signature: TypeSignatureValue,
      probability_budget_bits: float,
      maximum_expression_depth: int,
  ) -> None:
    self.grammar = grammar
    self.target_type_signature = target_type_signature
    self.probability_budget_bits = float(probability_budget_bits)
    self.maximum_expression_depth = int(maximum_expression_depth)

    self._generator: Iterator[tuple[
      DomainSpecificLanguageExpression, float]] | None = None

  def _create_generator(self) -> Iterator[
    tuple[DomainSpecificLanguageExpression, float]]:
    yielded_expressions: set[DomainSpecificLanguageExpression] = set()

    for expression, description_length_bits in self._enumerate_expressions_recursively(
        target_type_signature=self.target_type_signature,
        environment_type_signatures=(),
        remaining_budget_bits=self.probability_budget_bits,
        remaining_expression_depth=self.maximum_expression_depth,
    ):
      if expression in yielded_expressions:
        continue
      yielded_expressions.add(expression)
      yield expression, float(description_length_bits)

  def take_next_expressions(self, maximum_number_of_expressions: int) -> list[
    tuple[DomainSpecificLanguageExpression, float]]:
    maximum_number = max(0, int(maximum_number_of_expressions))
    if maximum_number == 0:
      return []

    if self._generator is None:
      self._generator = self._create_generator()

    expressions: list[tuple[DomainSpecificLanguageExpression, float]] = []
    while len(expressions) < maximum_number:
      try:
        expression, description_length_bits = next(self._generator)
      except StopIteration:
        # Restart enumeration from the beginning when exhausted.
        self._generator = self._create_generator()
        break
      expressions.append((expression, float(description_length_bits)))
    return expressions

  def _enumerate_expressions_recursively(
      self,
      target_type_signature: TypeSignatureValue,
      environment_type_signatures: tuple[TypeSignatureValue, ...],
      remaining_budget_bits: float,
      remaining_expression_depth: int,
  ) -> Iterator[tuple[DomainSpecificLanguageExpression, float]]:
    if remaining_budget_bits < 0.0:
      return
    if remaining_expression_depth < 0:
      return

    # Variable productions
    variable_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
      "variable")
    if remaining_expression_depth >= 0:
      for variable_index, variable_type_signature in enumerate(
          reversed(environment_type_signatures)):
        if variable_type_signature != target_type_signature:
          continue
        variable_index_penalty_bits = math.log2(float(variable_index) + 2.0)
        total_bits = variable_constructor_bits + variable_index_penalty_bits
        if total_bits <= remaining_budget_bits:
          yield VariableExpression(variable_index=variable_index), total_bits

    # Constant productions
    constant_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
      "constant")
    for constant_candidate in self.grammar.list_constant_candidates(
        target_type_signature):
      total_bits = constant_constructor_bits + constant_candidate.description_length_bits
      if total_bits <= remaining_budget_bits:
        yield ConstantExpression(value=constant_candidate.value), total_bits

    # Primitive call productions
    primitive_call_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
      "primitive_call")
    for primitive_name, primitive_definition in self.grammar.list_primitives_returning_type(
        target_type_signature):
      primitive_bits = primitive_definition.description_length_bits
      node_bits = primitive_call_constructor_bits + primitive_bits
      if node_bits > remaining_budget_bits:
        continue

      argument_type_signatures = primitive_definition.argument_type_signatures
      for argument_expressions, argument_bits in self._enumerate_argument_expression_lists(
          argument_type_signatures=argument_type_signatures,
          environment_type_signatures=environment_type_signatures,
          remaining_budget_bits=(remaining_budget_bits - node_bits),
          remaining_expression_depth=(remaining_expression_depth - 1),
      ):
        total_bits = node_bits + argument_bits
        if total_bits <= remaining_budget_bits:
          yield PrimitiveCallExpression(
            primitive_name=primitive_name,
            argument_expressions=argument_expressions,
          ), total_bits

    # If productions
    if_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
      "if")
    if if_constructor_bits <= remaining_budget_bits and remaining_expression_depth >= 1:
      condition_budget = remaining_budget_bits - if_constructor_bits
      for condition_expression, condition_bits in self._enumerate_expressions_recursively(
          target_type_signature=BOOLEAN_TYPE,
          environment_type_signatures=environment_type_signatures,
          remaining_budget_bits=condition_budget,
          remaining_expression_depth=remaining_expression_depth - 1,
      ):
        for then_expression, then_bits in self._enumerate_expressions_recursively(
            target_type_signature=target_type_signature,
            environment_type_signatures=environment_type_signatures,
            remaining_budget_bits=condition_budget - condition_bits,
            remaining_expression_depth=remaining_expression_depth - 1,
        ):
          for else_expression, else_bits in self._enumerate_expressions_recursively(
              target_type_signature=target_type_signature,
              environment_type_signatures=environment_type_signatures,
              remaining_budget_bits=condition_budget - condition_bits - then_bits,
              remaining_expression_depth=remaining_expression_depth - 1,
          ):
            total_bits = if_constructor_bits + condition_bits + then_bits + else_bits
            if total_bits <= remaining_budget_bits:
              yield IfExpression(
                condition_expression=condition_expression,
                then_expression=then_expression,
                else_expression=else_expression,
              ), total_bits

    # Lambda productions
    if isinstance(target_type_signature, FunctionType):
      lambda_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
        "lambda")
      if lambda_constructor_bits <= remaining_budget_bits and remaining_expression_depth >= 1:
        extended_environment = environment_type_signatures + (
          target_type_signature.argument_type,)
        for body_expression, body_bits in self._enumerate_expressions_recursively(
            target_type_signature=target_type_signature.return_type,
            environment_type_signatures=extended_environment,
            remaining_budget_bits=(
                remaining_budget_bits - lambda_constructor_bits),
            remaining_expression_depth=(remaining_expression_depth - 1),
        ):
          total_bits = lambda_constructor_bits + body_bits
          if total_bits <= remaining_budget_bits:
            yield LambdaExpression(
              parameter_type_signature=target_type_signature.argument_type,
              body_expression=body_expression,
            ), total_bits

    # Apply productions
    apply_constructor_bits = self.grammar.get_syntax_constructor_description_length_bits(
      "apply")
    if apply_constructor_bits <= remaining_budget_bits and remaining_expression_depth >= 1:
      for argument_type_signature in self.grammar.list_application_argument_types():
        function_type_signature = FunctionType(argument_type_signature,
                                               target_type_signature)

        for function_expression, function_bits in self._enumerate_expressions_recursively(
            target_type_signature=function_type_signature,
            environment_type_signatures=environment_type_signatures,
            remaining_budget_bits=(
                remaining_budget_bits - apply_constructor_bits),
            remaining_expression_depth=(remaining_expression_depth - 1),
        ):
          for argument_expression, argument_bits in self._enumerate_expressions_recursively(
              target_type_signature=argument_type_signature,
              environment_type_signatures=environment_type_signatures,
              remaining_budget_bits=(
                  remaining_budget_bits - apply_constructor_bits - function_bits),
              remaining_expression_depth=(remaining_expression_depth - 1),
          ):
            total_bits = apply_constructor_bits + function_bits + argument_bits
            if total_bits <= remaining_budget_bits:
              yield ApplyExpression(
                function_expression=function_expression,
                argument_expression=argument_expression,
              ), total_bits

  def _enumerate_argument_expression_lists(
      self,
      argument_type_signatures: tuple[TypeSignatureValue, ...],
      environment_type_signatures: tuple[TypeSignatureValue, ...],
      remaining_budget_bits: float,
      remaining_expression_depth: int,
  ) -> Iterator[tuple[tuple[DomainSpecificLanguageExpression, ...], float]]:
    if len(argument_type_signatures) == 0:
      yield (), 0.0
      return

    first_argument_type_signature = argument_type_signatures[0]
    remaining_argument_type_signatures = argument_type_signatures[1:]

    for first_expression, first_bits in self._enumerate_expressions_recursively(
        target_type_signature=first_argument_type_signature,
        environment_type_signatures=environment_type_signatures,
        remaining_budget_bits=remaining_budget_bits,
        remaining_expression_depth=remaining_expression_depth,
    ):
      remaining_budget_after_first = remaining_budget_bits - first_bits
      if remaining_budget_after_first < 0.0:
        continue

      for remaining_expressions, remaining_bits in self._enumerate_argument_expression_lists(
          argument_type_signatures=remaining_argument_type_signatures,
          environment_type_signatures=environment_type_signatures,
          remaining_budget_bits=remaining_budget_after_first,
          remaining_expression_depth=remaining_expression_depth,
      ):
        total_bits = first_bits + remaining_bits
        if total_bits <= remaining_budget_bits:
          yield (first_expression,) + remaining_expressions, total_bits


class EnumerativeTransformerSearchStrategy:
  """Enumerative DreamCoder-style transformer search.

  - Enumerates DSL expressions as transformers.
  - Each expression is annotated with its description length in bits, derived from the grammar.
  - The same interface can later be implemented by a sampling strategy (e.g., recognition network guidance).
  """

  def __init__(
      self,
      grammar: ProbabilisticProgramGrammar,
      target_type_signature: TypeSignatureValue = DISTRIBUTION_TYPE,
      probability_budget_bits: float = 20.0,
      maximum_expression_depth: int = 6,
  ) -> None:
    self.grammar = grammar
    self.target_type_signature = target_type_signature
    self.probability_budget_bits = float(probability_budget_bits)
    self.maximum_expression_depth = int(maximum_expression_depth)

    self._enumerator = EnumerativeExpressionEnumerator(
      grammar=self.grammar,
      target_type_signature=self.target_type_signature,
      probability_budget_bits=self.probability_budget_bits,
      maximum_expression_depth=self.maximum_expression_depth,
    )

  def propose_transformer_expressions(
      self,
      evaluation_context: DomainSpecificLanguageEvaluationContext,
      maximum_number_of_expressions: int,
  ) -> list[TransformerExpressionCandidate]:
    del evaluation_context  # placeholder for future conditioning

    proposed = self._enumerator.take_next_expressions(
      maximum_number_of_expressions=maximum_number_of_expressions)
    candidates: list[TransformerExpressionCandidate] = []
    for expression, description_length_bits in proposed:
      log_probability_base_two = -float(description_length_bits)
      candidates.append(
        TransformerExpressionCandidate(
          transformer_expression=expression,
          description_length_bits=float(description_length_bits),
          log_probability_base_two=float(log_probability_base_two),
        )
      )
    return candidates
