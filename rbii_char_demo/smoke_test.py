from __future__ import annotations

import math
import os
from dataclasses import dataclass

from configuration import ResourceBoundedIncrementalInductionConfiguration
from resource_bounded_incremental_induction import ResourceBoundedIncrementalInduction


@dataclass(frozen=True)
class SimpleCharacterVocabulary:
    index_to_character: tuple[str, ...]
    character_to_index: dict[str, int]
    size: int

    @staticmethod
    def from_text(text: str, extra_characters: str = "") -> "SimpleCharacterVocabulary":
        unique_characters: list[str] = []
        seen: set[str] = set()

        for character in (extra_characters + text):
            if character in seen:
                continue
            seen.add(character)
            unique_characters.append(character)

        if "?" not in seen:
            unique_characters.append("?")

        index_to_character = tuple(unique_characters)
        character_to_index = {character: index for index, character in enumerate(index_to_character)}
        return SimpleCharacterVocabulary(
            index_to_character=index_to_character,
            character_to_index=character_to_index,
            size=len(index_to_character),
        )

    def encode_text(self, text: str) -> list[int]:
        unknown_index = self.character_to_index.get("?", 0)
        return [self.character_to_index.get(character, unknown_index) for character in text]

    def decode_indices(self, indices: list[int]) -> str:
        return "".join(self.index_to_character[index] for index in indices)


def build_smoke_test_stream() -> str:
    # Simple regime switch: A (predict 'a'), then B (predict 'b'), then A again.
    # This is intentionally easy so you can verify the end-to-end plumbing quickly.
    return ("a" * 200) + ("b" * 200) + ("a" * 200)


def describe_predictor_program(predictor_program: object) -> str:
    expression = getattr(predictor_program, "expression", None)
    if expression is None:
        return predictor_program.__class__.__name__
    return repr(expression)


def main() -> None:
    text_stream = build_smoke_test_stream()
    character_vocabulary = SimpleCharacterVocabulary.from_text(text_stream, extra_characters="ab")
    encoded_stream = character_vocabulary.encode_text(text_stream)

    configuration = ResourceBoundedIncrementalInductionConfiguration(
        pool_capacity=6,
        exploration_transformer_executions_per_step=4,
        transformer_search_probability_budget_bits=18,
        transformer_search_maximum_expression_depth=6,
        transformer_search_maximum_expressions_per_step=64,
        validation_window_length=128,
        candidate_buffer_capacity=32,
        candidate_validation_interval=32,
        freeze_evaluation_interval=32,
        probability_floor=1e-3,
        detectability_slack_bits=4.0,
        maximum_recalled_programs_per_step=4,
        maximum_worker_threads=os.cpu_count(),
    )

    resource_bounded_incremental_induction_system = ResourceBoundedIncrementalInduction(
        character_vocabulary=character_vocabulary,
        configuration=configuration,
    )

    primitive_names = resource_bounded_incremental_induction_system.primitive_library.list_primitive_names()
    assert all(
        ("bigram" not in name and "digram" not in name and "trigram" not in name)
        for name in primitive_names
    ), "The primitive library contains n-gram primitives, which this demo is supposed to avoid."

    loss_bits_by_step: list[float] = []

    previous_active_program_identifiers = {
        entry.program_identifier for entry in resource_bounded_incremental_induction_system.active_pool
    }

    for time_step_index, observed_character_index in enumerate(encoded_stream, start=1):
        predicted_distribution = resource_bounded_incremental_induction_system.step(
            observed_character_index=observed_character_index
        )
        del predicted_distribution  # prediction is available if you want to inspect it

        loss_bits = float(resource_bounded_incremental_induction_system.last_loss_bits or 0.0)
        loss_bits_by_step.append(loss_bits)

        current_active_program_identifiers = {
            entry.program_identifier for entry in resource_bounded_incremental_induction_system.active_pool
        }
        if current_active_program_identifiers != previous_active_program_identifiers:
            inserted_program_identifiers = sorted(current_active_program_identifiers - previous_active_program_identifiers)
            if inserted_program_identifiers:
                print(f"[step {time_step_index}] inserted predictors: {len(inserted_program_identifiers)}")
                for program_identifier in inserted_program_identifiers[:6]:
                    matching_entry = next(
                        entry
                        for entry in resource_bounded_incremental_induction_system.active_pool
                        if entry.program_identifier == program_identifier
                    )
                    print("  program_identifier_prefix:", program_identifier[:12])
                    print("  description_length_bits:", round(float(matching_entry.description_length_bits), 3))
                    print("  predictor:", describe_predictor_program(matching_entry.predictor_program))

            previous_active_program_identifiers = current_active_program_identifiers

        if time_step_index % 50 == 0:
            recent_losses = loss_bits_by_step[-50:]
            recent_average_loss_bits = sum(recent_losses) / float(len(recent_losses))
            uniform_loss_bits_per_character = math.log2(float(character_vocabulary.size))
            frozen_program_count = len(resource_bounded_incremental_induction_system.frozen_store.list_program_identifiers())

            print(
                f"[step {time_step_index}] "
                f"recent_average_loss_bits_per_character={recent_average_loss_bits:.3f} "
                f"(uniform={uniform_loss_bits_per_character:.3f}) "
                f"active_pool={len(resource_bounded_incremental_induction_system.active_pool)} "
                f"frozen_store={frozen_program_count}"
            )

    average_loss_bits_per_character = sum(loss_bits_by_step) / float(len(loss_bits_by_step))
    uniform_loss_bits_per_character = math.log2(float(character_vocabulary.size))
    average_compression_gain_bits_per_character = uniform_loss_bits_per_character - average_loss_bits_per_character

    print("\n=== smoke test summary ===")
    print("total_characters:", len(encoded_stream))
    print("vocabulary_size:", character_vocabulary.size)
    print("average_loss_bits_per_character:", round(float(average_loss_bits_per_character), 6))
    print("uniform_loss_bits_per_character:", round(float(uniform_loss_bits_per_character), 6))
    print("average_compression_gain_bits_per_character:", round(float(average_compression_gain_bits_per_character), 6))
    print("frozen_programs:", len(resource_bounded_incremental_induction_system.frozen_store.list_program_identifiers()))


if __name__ == "__main__":
    main()
