from __future__ import annotations

import json
from pathlib import Path

from character_vocabulary import CharacterVocabulary
from configuration import ResourceBoundedIncrementalInductionConfiguration
from freezing_policies import IncumbentRunLengthFreezePolicy
from memory_mechanisms import CharacterHistoryMemoryMechanism
from metrics import (
    compute_cumulative_sum,
    compute_cumulative_compression_gain_bits,
    compute_reacquisition_measurements,
    compute_reference_loss_bits_for_scenario,
)
from newborn_weight_policies import PriorConsistentNewbornWeightAssignmentPolicy
from primitive_library import create_default_primitive_library
from resource_bounded_incremental_induction import ResourceBoundedIncrementalInductionSystem
from test_scenarios import (
    build_scenario_a_context_switching,
    build_scenario_b_compositional_curriculum,
    build_scenario_c_rare_glimpses,
    build_scenario_d_non_recurrent_drift,
)
from plotting import (
    plot_cumulative_compression_gain,
    plot_cumulative_loss,
    plot_reacquisition_delay,
    plot_reacquisition_excess_loss,
)
from transformer_programs import (
    BigramPredictorTransformerProgram,
    DigitCyclePredictorTransformerProgram,
    DigitCycleStepEditTransformerProgram,
    MixtureOfRecalledProgramsTransformerProgram,
    RecallFrozenProgramTransformerProgram,
    TrigramPredictorTransformerProgram,
    UniformPredictorTransformerProgram,
)


def run_scenario(scenario, outputs_directory: Path) -> None:
    vocabulary = CharacterVocabulary.from_text(scenario.stream_text)
    character_indices = vocabulary.encode_text(scenario.stream_text)

    configuration = ResourceBoundedIncrementalInductionConfiguration(
        pool_capacity=8,
        exploration_transformer_executions_per_step=3,
        validation_window_length=256,
        candidate_buffer_capacity=32,
        candidate_validation_interval=32,
        freeze_evaluation_interval=32,
        probability_floor=1e-3,
        detectability_slack_bits=8.0,
        reacquisition_tolerance_bits_per_character=0.25,
        maximum_recalled_programs_per_transformer=4,
        maximum_worker_threads=None,
    )

    primitive_library = create_default_primitive_library()
    memory_mechanism = CharacterHistoryMemoryMechanism(
        maximum_context_length=16,
        recall_key_length=4,
        smoothing_alpha=0.5,
        maximum_program_identifiers_per_key=8,
    )
    freezing_policy = IncumbentRunLengthFreezePolicy(
        minimum_incumbent_run_length=256,
        minimum_average_gain_bits_per_character=0.1,
    )
    newborn_weight_assignment_policy = PriorConsistentNewbornWeightAssignmentPolicy()

    transformers = [
        UniformPredictorTransformerProgram(),
        BigramPredictorTransformerProgram(),
        TrigramPredictorTransformerProgram(),
        RecallFrozenProgramTransformerProgram(),
        MixtureOfRecalledProgramsTransformerProgram(),
        DigitCycleStepEditTransformerProgram(step_change=1),
        DigitCycleStepEditTransformerProgram(step_change=-1),
    ]
    for step_size in range(1, 10):
        transformers.append(DigitCyclePredictorTransformerProgram(step_size=step_size))

    system = ResourceBoundedIncrementalInductionSystem(
        configuration=configuration,
        primitive_library=primitive_library,
        transformer_programs=transformers,
        memory_mechanism=memory_mechanism,
        freezing_policy=freezing_policy,
        newborn_weight_assignment_policy=newborn_weight_assignment_policy,
        random_seed=0,
    )
    run_result = system.run(character_indices=character_indices, character_vocabulary=vocabulary)

    reference_loss_bits = compute_reference_loss_bits_for_scenario(scenario, vocabulary)
    cumulative_algorithm_loss = compute_cumulative_sum(run_result.per_step_algorithm_loss_bits)[-1]
    cumulative_baseline_loss = compute_cumulative_sum(run_result.per_step_baseline_loss_bits)[-1]
    cumulative_gain_bits = cumulative_baseline_loss - cumulative_algorithm_loss

    reacquisition_measurements = compute_reacquisition_measurements(
        scenario=scenario,
        per_step_algorithm_loss_bits=run_result.per_step_algorithm_loss_bits,
        per_step_reference_loss_bits=reference_loss_bits,
        tolerance_bits_per_character=configuration.reacquisition_tolerance_bits_per_character,
    )

    scenario_outputs_directory = outputs_directory / scenario.scenario_name
    scenario_outputs_directory.mkdir(parents=True, exist_ok=True)

    plot_cumulative_loss(
        scenario_outputs_directory / f"{scenario.scenario_name}_cumulative_loss.png",
        per_step_algorithm_loss_bits=run_result.per_step_algorithm_loss_bits,
        per_step_baseline_loss_bits=run_result.per_step_baseline_loss_bits,
        per_step_reference_loss_bits=reference_loss_bits,
        title=f"{scenario.scenario_name}: cumulative loss",
    )

    plot_cumulative_compression_gain(
        scenario_outputs_directory / f"{scenario.scenario_name}_cumulative_compression_gain.png",
        per_step_algorithm_loss_bits=run_result.per_step_algorithm_loss_bits,
        per_step_baseline_loss_bits=run_result.per_step_baseline_loss_bits,
        title=f"{scenario.scenario_name}: cumulative compression gain",
    )

    if len(reacquisition_measurements) > 0:
        plot_reacquisition_delay(
            scenario_outputs_directory / f"{scenario.scenario_name}_reacquisition_delay.png",
            measurements=reacquisition_measurements,
            title=f"{scenario.scenario_name}: reacquisition delay",
        )
        plot_reacquisition_excess_loss(
            scenario_outputs_directory / f"{scenario.scenario_name}_reacquisition_excess_loss.png",
            measurements=reacquisition_measurements,
            title=f"{scenario.scenario_name}: reacquisition excess loss",
        )

    metrics_payload = {
        "scenario_name": scenario.scenario_name,
        "stream_length_characters": len(scenario.stream_text),
        "final_cumulative_algorithm_loss_bits": cumulative_algorithm_loss,
        "final_cumulative_baseline_loss_bits": cumulative_baseline_loss,
        "final_cumulative_compression_gain_bits": cumulative_gain_bits,
        "number_of_episodes": len(scenario.episodes),
        "reacquisition_measurements": [
            {
                "task_name": m.task_name,
                "return_index": m.return_index,
                "episode_start_index": m.episode_start_index,
                "episode_end_index": m.episode_end_index,
                "episode_length": m.episode_length,
                "reacquisition_delay": m.reacquisition_delay,
                "reacquisition_excess_loss_bits": m.reacquisition_excess_loss_bits,
            }
            for m in reacquisition_measurements
        ],
    }
    (scenario_outputs_directory / f"{scenario.scenario_name}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    repository_directory = Path(__file__).resolve().parent
    data_directory = repository_directory / "data"
    outputs_directory = repository_directory / "outputs"
    outputs_directory.mkdir(parents=True, exist_ok=True)

    scenarios = [
        build_scenario_a_context_switching(data_directory=data_directory),
        build_scenario_b_compositional_curriculum(),
        build_scenario_c_rare_glimpses(data_directory=data_directory),
        build_scenario_d_non_recurrent_drift(),
    ]

    for scenario in scenarios:
        print(f"Running {scenario.scenario_name} (length={len(scenario.stream_text)})...")
        run_scenario(scenario, outputs_directory=outputs_directory)

    print(f"Done. Outputs written to: {outputs_directory}")


if __name__ == "__main__":
    main()
