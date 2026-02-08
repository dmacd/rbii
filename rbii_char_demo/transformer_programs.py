from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain_specific_language import (
    constant_expression,
    frozen_program_call_expression,
    primitive_call_expression,
)
from predictor_programs import domain_specific_language_predictor_program
from primitive_library import primitive_library
from memory_mechanisms import memory_mechanism, memory_state_protocol


@dataclass(frozen=True)
class transformer_candidate:
    predictor_program: domain_specific_language_predictor_program
    transformer_description_length_bits: int
    candidate_signature: str


class transformer_program(Protocol):
    description_length_bits: int

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: "frozen_program_store_protocol",
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate | None: ...


class frozen_program_store_protocol(Protocol):
    def get_program(self, program_identifier: str): ...


@dataclass(frozen=True)
class uniform_predictor_transformer_program:
    description_length_bits: int = 2

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate:
        expression = primitive_call_expression("uniform_character_distribution", ())
        program = domain_specific_language_predictor_program(expression=expression)
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=repr(expression),
        )


@dataclass(frozen=True)
class bigram_predictor_transformer_program:
    description_length_bits: int = 3

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate:
        expression = primitive_call_expression("bigram_character_distribution", ())
        program = domain_specific_language_predictor_program(expression=expression)
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=repr(expression),
        )


@dataclass(frozen=True)
class trigram_predictor_transformer_program:
    description_length_bits: int = 4

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate:
        expression = primitive_call_expression("trigram_character_distribution", ())
        program = domain_specific_language_predictor_program(expression=expression)
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=repr(expression),
        )


@dataclass(frozen=True)
class digit_cycle_predictor_transformer_program:
    step_size: int
    description_length_bits: int = 5

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate | None:
        # Only propose if the recall key context looks digit-heavy.
        recall_key = memory_mechanism.build_recall_key(memory_state)
        recent_characters = []
        for index in recall_key:
            if index < 0:
                continue
            recent_characters.append(memory_state.character_vocabulary.index_to_character[index])
        if len(recent_characters) == 0:
            return None
        digit_count = sum(1 for character in recent_characters if character.isdigit())
        if digit_count < max(1, len(recent_characters) // 2):
            return None

        expression = primitive_call_expression(
            "digit_cycle_character_distribution",
            (constant_expression(int(self.step_size)),),
        )
        program = domain_specific_language_predictor_program(expression=expression)
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=repr(expression),
        )


@dataclass(frozen=True)
class recall_frozen_program_transformer_program:
    description_length_bits: int = 2

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate | None:
        recall_key = memory_mechanism.build_recall_key(memory_state)
        recalled_identifiers = memory_mechanism.recall_program_identifiers(
            memory_state=memory_state,
            recall_key=recall_key,
            maximum_number=1,
        )
        if len(recalled_identifiers) == 0:
            return None

        identifier = recalled_identifiers[0]
        expression = frozen_program_call_expression(identifier)
        program = domain_specific_language_predictor_program(expression=expression)
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=f"recall:{identifier}",
        )


@dataclass(frozen=True)
class mixture_of_recalled_programs_transformer_program:
    description_length_bits: int = 3
    first_weight: float = 0.5

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate | None:
        recall_key = memory_mechanism.build_recall_key(memory_state)
        recalled_identifiers = memory_mechanism.recall_program_identifiers(
            memory_state=memory_state,
            recall_key=recall_key,
            maximum_number=2,
        )
        if len(recalled_identifiers) < 2:
            return None

        first_identifier, second_identifier = recalled_identifiers[0], recalled_identifiers[1]
        expression = primitive_call_expression(
            "mixture_of_two_character_distributions",
            (
                frozen_program_call_expression(first_identifier),
                frozen_program_call_expression(second_identifier),
                constant_expression(float(self.first_weight)),
            ),
        )
        program = domain_specific_language_predictor_program(expression=expression)
        signature = f"mixture_recall:{first_identifier}:{second_identifier}:{self.first_weight}"
        return transformer_candidate(
            predictor_program=program,
            transformer_description_length_bits=self.description_length_bits,
            candidate_signature=signature,
        )


@dataclass(frozen=True)
class digit_cycle_step_edit_transformer_program:
    step_change: int
    description_length_bits: int = 3

    def generate_candidate(
        self,
        memory_state: memory_state_protocol,
        frozen_store: frozen_program_store_protocol,
        primitive_library: primitive_library,
        memory_mechanism: memory_mechanism,
    ) -> transformer_candidate | None:
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
                and getattr(expression, "primitive_name") == "digit_cycle_character_distribution"
                and hasattr(expression, "argument_expressions")
                and len(getattr(expression, "argument_expressions")) == 1
            ):
                argument_expression = getattr(expression, "argument_expressions")[0]
                if not hasattr(argument_expression, "value"):
                    continue
                current_step_size = int(getattr(argument_expression, "value"))
                new_step_size = (current_step_size + int(self.step_change)) % 10
                new_expression = primitive_call_expression(
                    "digit_cycle_character_distribution",
                    (constant_expression(int(new_step_size)),),
                )
                new_program = domain_specific_language_predictor_program(expression=new_expression)
                signature = f"digit_cycle_edit:{identifier}:{current_step_size}->{new_step_size}"
                return transformer_candidate(
                    predictor_program=new_program,
                    transformer_description_length_bits=self.description_length_bits,
                    candidate_signature=signature,
                )
        return None
