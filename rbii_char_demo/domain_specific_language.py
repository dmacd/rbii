from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as numpy

from character_vocabulary import character_vocabulary
from primitive_library import primitive_library


@dataclass(frozen=True)
class prediction_features:
    recent_character_indices: tuple[int, ...]
    bigram_probability_distribution: numpy.ndarray
    trigram_probability_distribution: numpy.ndarray


@dataclass(frozen=True)
class domain_specific_language_evaluation_context:
    character_vocabulary: character_vocabulary
    prediction_features: prediction_features
    probability_floor: float


class frozen_program_store(Protocol):
    def get_program(self, program_identifier: str) -> Any: ...


class domain_specific_language_expression(Protocol):
    def evaluate(
        self,
        evaluation_context: domain_specific_language_evaluation_context,
        primitive_library: primitive_library,
        frozen_store: frozen_program_store,
    ) -> Any: ...


@dataclass(frozen=True)
class constant_expression:
    value: Any

    def evaluate(
        self,
        evaluation_context: domain_specific_language_evaluation_context,
        primitive_library: primitive_library,
        frozen_store: frozen_program_store,
    ) -> Any:
        return self.value


@dataclass(frozen=True)
class primitive_call_expression:
    primitive_name: str
    argument_expressions: tuple[domain_specific_language_expression, ...]

    def evaluate(
        self,
        evaluation_context: domain_specific_language_evaluation_context,
        primitive_library: primitive_library,
        frozen_store: frozen_program_store,
    ) -> Any:
        definition = primitive_library.get_definition(self.primitive_name)
        evaluated_arguments = [
            argument_expression.evaluate(evaluation_context, primitive_library, frozen_store)
            for argument_expression in self.argument_expressions
        ]
        return definition.callable_function(evaluation_context, *evaluated_arguments)


@dataclass(frozen=True)
class frozen_program_call_expression:
    frozen_program_identifier: str

    def evaluate(
        self,
        evaluation_context: domain_specific_language_evaluation_context,
        primitive_library: primitive_library,
        frozen_store: frozen_program_store,
    ) -> Any:
        program = frozen_store.get_program(self.frozen_program_identifier)
        distribution = program.predict_character_distribution(
            evaluation_context=evaluation_context,
            primitive_library=primitive_library,
            frozen_store=frozen_store,
        )
        return distribution
