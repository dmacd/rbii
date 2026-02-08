from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain_specific_language import (
  ConstantExpression,
  FrozenProgramCallExpression,
  PrimitiveCallExpression,
)
from predictor_programs import DomainSpecificLanguagePredictorProgram
from primitive_library import PrimitiveLibrary
from memory_mechanisms import MemoryMechanism, MemoryStateProtocol


@dataclass(frozen=True)
class TransformerCandidate:
  predictor_program: DomainSpecificLanguagePredictorProgram
  transformer_description_length_bits: int
  candidate_signature: str


class TransformerProgram(Protocol):
  description_length_bits: int

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: "FrozenProgramStoreProtocol",
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate | None: ...


class FrozenProgramStoreProtocol(Protocol):
  def get_program(self, program_identifier: str): ...


@dataclass(frozen=True)
class UniformPredictorTransformerProgram:
  description_length_bits: int = 2

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate:
    expression = PrimitiveCallExpression("uniform_character_distribution", ())
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=repr(expression),
    )


@dataclass(frozen=True)
class BigramPredictorTransformerProgram:
  description_length_bits: int = 3

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate:
    expression = PrimitiveCallExpression("bigram_character_distribution", ())
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=repr(expression),
    )


@dataclass(frozen=True)
class TrigramPredictorTransformerProgram:
  description_length_bits: int = 4

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate:
    expression = PrimitiveCallExpression("trigram_character_distribution", ())
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=repr(expression),
    )


@dataclass(frozen=True)
class DigitCyclePredictorTransformerProgram:
  step_size: int
  description_length_bits: int = 5

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate | None:
    # Only propose if the recall key context looks digit-heavy.
    recall_key = memory_mechanism.build_recall_key(memory_state)
    recent_characters = []
    for index in recall_key:
      if index < 0:
        continue
      recent_characters.append(
        memory_state.character_vocabulary.index_to_character[index])
    if len(recent_characters) == 0:
      return None
    digit_count = sum(
      1 for character in recent_characters if character.isdigit())
    if digit_count < max(1, len(recent_characters) // 2):
      return None

    expression = PrimitiveCallExpression(
      "digit_cycle_character_distribution",
      (ConstantExpression(int(self.step_size)),),
    )
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=repr(expression),
    )


@dataclass(frozen=True)
class RecallFrozenProgramTransformerProgram:
  description_length_bits: int = 2

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate | None:
    recall_key = memory_mechanism.build_recall_key(memory_state)
    recalled_identifiers = memory_mechanism.recall_program_identifiers(
      memory_state=memory_state,
      recall_key=recall_key,
      maximum_number=1,
    )
    if len(recalled_identifiers) == 0:
      return None

    identifier = recalled_identifiers[0]
    expression = FrozenProgramCallExpression(identifier)
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=f"recall:{identifier}",
    )


@dataclass(frozen=True)
class MixtureOfRecalledProgramsTransformerProgram:
  description_length_bits: int = 3
  first_weight: float = 0.5

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate | None:
    recall_key = memory_mechanism.build_recall_key(memory_state)
    recalled_identifiers = memory_mechanism.recall_program_identifiers(
      memory_state=memory_state,
      recall_key=recall_key,
      maximum_number=2,
    )
    if len(recalled_identifiers) < 2:
      return None

    first_identifier, second_identifier = recalled_identifiers[0], \
    recalled_identifiers[1]
    expression = PrimitiveCallExpression(
      "mixture_of_two_character_distributions",
      (
        FrozenProgramCallExpression(first_identifier),
        FrozenProgramCallExpression(second_identifier),
        ConstantExpression(float(self.first_weight)),
      ),
    )
    program = DomainSpecificLanguagePredictorProgram(expression=expression)
    signature = f"mixture_recall:{first_identifier}:{second_identifier}:{self.first_weight}"
    return TransformerCandidate(
      predictor_program=program,
      transformer_description_length_bits=self.description_length_bits,
      candidate_signature=signature,
    )


@dataclass(frozen=True)
class DigitCycleStepEditTransformerProgram:
  step_change: int
  description_length_bits: int = 3

  def generate_candidate(
      self,
      memory_state: MemoryStateProtocol,
      frozen_store: FrozenProgramStoreProtocol,
      primitive_library: PrimitiveLibrary,
      memory_mechanism: MemoryMechanism,
  ) -> TransformerCandidate | None:
    # Attempt to find a recalled digit-cycle predictor and patch its step size.
    recall_key = memory_mechanism.build_recall_key(memory_state)
    recalled_identifiers = memory_mechanism.recall_program_identifiers(
      memory_state=memory_state,
      recall_key=recall_key,
      maximum_number=4,
    )
    for identifier in recalled_identifiers:
      program = frozen_store.get_program(identifier)
      if not hasattr(program, "expression"):
        continue
      expression = getattr(program, "expression")
      if (
          hasattr(expression, "primitive_name")
          and getattr(expression,
                      "primitive_name") == "digit_cycle_character_distribution"
          and hasattr(expression, "argument_expressions")
          and len(getattr(expression, "argument_expressions")) == 1
      ):
        argument_expression = getattr(expression, "argument_expressions")[0]
        if not hasattr(argument_expression, "value"):
          continue
        current_step_size = int(getattr(argument_expression, "value"))
        new_step_size = (current_step_size + int(self.step_change)) % 10
        new_expression = PrimitiveCallExpression(
          "digit_cycle_character_distribution",
          (ConstantExpression(int(new_step_size)),),
        )
        new_program = DomainSpecificLanguagePredictorProgram(
          expression=new_expression)
        signature = f"digit_cycle_edit:{identifier}:{current_step_size}->{new_step_size}"
        return TransformerCandidate(
          predictor_program=new_program,
          transformer_description_length_bits=self.description_length_bits,
          candidate_signature=signature,
        )
    return None
