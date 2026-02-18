from __future__ import annotations

import json
import math
import inspect
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from character_vocabulary import CharacterVocabulary
from configuration import ResourceBoundedIncrementalInductionConfiguration
from memory_mechanisms import CharacterHistoryMemoryMechanism
from metrics import (
  compute_cumulative_sum,
  compute_cumulative_compression_gain_bits,
  compute_reacquisition_measurements,
  compute_reference_loss_bits_for_scenario,
)
from primitive_library import create_default_primitive_library
from resource_bounded_incremental_induction import \
  ResourceBoundedIncrementalInduction
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
from predictor_programs import DomainSpecificLanguagePredictorProgram
from domain_specific_language import (
  ApplyExpression,
  ConstantExpression,
  FrozenProgramCallExpression,
  IfExpression,
  LambdaExpression,
  PrimitiveCallExpression,
  VariableExpression,
)
from tqdm.auto import tqdm

try:
  from clearml import Task
except ImportError:
  Task = None
# from transformer_programs import (
#   BigramPredictorTransformerProgram,
#   DigitCyclePredictorTransformerProgram,
#   DigitCycleStepEditTransformerProgram,
#   MixtureOfRecalledProgramsTransformerProgram,
#   RecallFrozenProgramTransformerProgram,
#   TrigramPredictorTransformerProgram,
#   UniformPredictorTransformerProgram,
# )


@dataclass(frozen=True)
class ScenarioRunResult:
  per_step_algorithm_loss_bits: list[float]
  per_step_baseline_loss_bits: list[float]


def _report_text_channel(
    clearml_logger: Any | None,
    title: str,
    series: str,
    step: int,
    text_payload: str,
) -> None:
  if clearml_logger is None:
    return

  try:
    clearml_logger.report_media(
      title=title,
      series=series,
      iteration=int(step),
      stream=StringIO(text_payload),
      file_extension=".txt",
    )
    return
  except Exception:
    pass

  clearml_logger.report_text(
    f"[{title}][{series}][step={int(step)}]\n{text_payload}"
  )


def _describe_predictor_program(predictor_program: object) -> str:
  expression = getattr(predictor_program, "expression", None)
  if expression is None:
    return predictor_program.__class__.__name__
  return repr(expression)


def _format_constant_value_as_lisp(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if value is None:
    return "nil"
  if isinstance(value, str):
    return json.dumps(value)
  if isinstance(value, tuple):
    return "(list " + " ".join(_format_constant_value_as_lisp(v)
                               for v in value) + ")"
  return str(value)


def _format_expression_as_lisp(expression: Any) -> str:
  if isinstance(expression, ConstantExpression):
    return _format_constant_value_as_lisp(expression.value)
  if isinstance(expression, VariableExpression):
    return f"$v{int(expression.variable_index)}"
  if isinstance(expression, LambdaExpression):
    parameter_type = expression.parameter_type_signature
    parameter_type_text = "any" if parameter_type is None else str(
      parameter_type)
    body_text = _format_expression_as_lisp(expression.body_expression)
    return f"(lambda ({parameter_type_text}) {body_text})"
  if isinstance(expression, ApplyExpression):
    function_text = _format_expression_as_lisp(expression.function_expression)
    argument_text = _format_expression_as_lisp(expression.argument_expression)
    return f"(apply {function_text} {argument_text})"
  if isinstance(expression, IfExpression):
    condition_text = _format_expression_as_lisp(expression.condition_expression)
    then_text = _format_expression_as_lisp(expression.then_expression)
    else_text = _format_expression_as_lisp(expression.else_expression)
    return f"(if {condition_text} {then_text} {else_text})"
  if isinstance(expression, PrimitiveCallExpression):
    argument_texts = [
      _format_expression_as_lisp(argument_expression)
      for argument_expression in expression.argument_expressions
    ]
    if len(argument_texts) == 0:
      return f"({expression.primitive_name})"
    return f"({expression.primitive_name} {' '.join(argument_texts)})"
  if isinstance(expression, FrozenProgramCallExpression):
    return f"(recall-frozen {expression.frozen_program_identifier[:12]})"
  return repr(expression)


def _format_predictor_program_as_lisp(predictor_program: object) -> str:
  if isinstance(predictor_program, DomainSpecificLanguagePredictorProgram):
    return _format_expression_as_lisp(predictor_program.expression)
  expression = getattr(predictor_program, "expression", None)
  if expression is not None:
    return _format_expression_as_lisp(expression)
  return predictor_program.__class__.__name__


def _write_dsl_documentation_file(
    primitive_library: Any,
    output_file_path: Path,
) -> None:
  output_file_path.parent.mkdir(parents=True, exist_ok=True)
  lines: list[str] = [
    ";; RBII DSL documentation (Lisp style)",
    ";; Each primitive is called as: (primitive_name arg1 arg2 ...)",
    "",
  ]

  for primitive_name in primitive_library.list_primitive_names():
    primitive_definition = primitive_library.get_definition(primitive_name)
    python_signature = inspect.signature(primitive_definition.callable_function)
    parameter_names = list(python_signature.parameters.keys())[3:]
    argument_types = primitive_definition.argument_type_signatures
    if len(parameter_names) != len(argument_types):
      parameter_names = [f"arg{i + 1}" for i in range(len(argument_types))]

    lisp_call_parts = [primitive_name]
    argument_lines: list[str] = []
    for parameter_name, argument_type in zip(parameter_names, argument_types):
      lisp_call_parts.append(f"<{parameter_name}>")
      argument_lines.append(f"    ({parameter_name} {argument_type})")

    lines.append(f"(primitive {primitive_name}")
    lines.append(f"  (call-syntax ({' '.join(lisp_call_parts)}))")
    lines.append("  (arguments")
    if len(argument_lines) == 0:
      lines.append("    ()")
    else:
      lines.extend(argument_lines)
    lines.append("  )")
    lines.append(
      f"  (returns {primitive_definition.return_type_signature})")
    lines.append(
      f"  (description-length-bits {primitive_definition.description_length_bits:.6f})"
    )
    lines.append(")")
    lines.append("")

  output_file_path.write_text("\n".join(lines), encoding="utf-8")


def _write_program_snapshot_file(
    snapshot_directory: Path,
    step_index: int,
    scenario_name: str,
    system: ResourceBoundedIncrementalInduction,
) -> None:
  snapshot_directory.mkdir(parents=True, exist_ok=True)
  snapshot_file_path = snapshot_directory / f"step_{step_index:06d}.txt"

  active_entries = sorted(system.active_pool,
                          key=lambda entry: float(entry.log_weight_base_two),
                          reverse=True)
  candidate_entries = sorted(
    system.candidate_buffer,
    key=lambda entry: float(entry.minimum_description_length_score_bits),
  )

  lines: list[str] = [
    f"scenario={scenario_name}",
    f"step={step_index}",
    f"active_pool_size={len(active_entries)}",
    f"candidate_buffer_size={len(candidate_entries)}",
    "",
    ";; Active predictors",
  ]

  for rank, entry in enumerate(active_entries, start=1):
    lines.append(f"(active_predictor {rank}")
    lines.append(f"  (program_identifier {entry.program_identifier})")
    lines.append(f"  (description_length_bits {float(entry.description_length_bits):.6f})")
    lines.append(f"  (log_weight_base_two {float(entry.log_weight_base_two):.6f})")
    lines.append("  (predictor_expression")
    lines.append(f"    {_format_predictor_program_as_lisp(entry.predictor_program)}")
    lines.append("  )")
    lines.append(")")
    lines.append("")

  lines.append(";; Transformer candidates and generated predictors")
  for rank, entry in enumerate(candidate_entries, start=1):
    expression_lisp = _format_predictor_program_as_lisp(entry.predictor_program)
    lines.append(f"(candidate_transformer {rank}")
    lines.append(f"  (program_identifier {entry.program_identifier})")
    lines.append(f"  (description_length_bits {float(entry.description_length_bits):.6f})")
    lines.append(
      f"  (minimum_description_length_score_bits {float(entry.minimum_description_length_score_bits):.6f})"
    )
    lines.append("  (transformer_expression")
    lines.append(f"    {expression_lisp}")
    lines.append("  )")
    lines.append("  (generated_predictor_expression")
    lines.append(f"    {expression_lisp}")
    lines.append("  )")
    lines.append(")")
    lines.append("")

  snapshot_file_path.write_text("\n".join(lines), encoding="utf-8")


def _format_active_pool_snapshot(
    system: ResourceBoundedIncrementalInduction,
    step_index: int,
    scenario_name: str,
) -> str:
  sorted_pool_entries = sorted(
    system.active_pool,
    key=lambda entry: float(entry.log_weight_base_two),
    reverse=True,
  )

  lines = [
    f"scenario={scenario_name}",
    f"step={step_index}",
    f"active_pool_size={len(sorted_pool_entries)}",
    "active_pool_entries:",
  ]
  for rank, entry in enumerate(sorted_pool_entries, start=1):
    lines.append(
      " | ".join([
        f"rank={rank}",
        f"program_id_prefix={entry.program_identifier[:12]}",
        f"log_weight_base_two={float(entry.log_weight_base_two):.3f}",
        f"description_length_bits={float(entry.description_length_bits):.3f}",
        f"predictor={_describe_predictor_program(entry.predictor_program)}",
      ]))
  return "\n".join(lines)


def _format_frozen_store_snapshot(
    system: ResourceBoundedIncrementalInduction,
    step_index: int,
    scenario_name: str,
    frozen_program_identifiers: tuple[str, ...],
) -> str:
  lines = [
    f"scenario={scenario_name}",
    f"step={step_index}",
    f"frozen_store_size={len(frozen_program_identifiers)}",
    "frozen_program_identifier_prefixes:",
  ]
  lines.extend(program_identifier[:12]
               for program_identifier in frozen_program_identifiers)
  return "\n".join(lines)


def run_scenario_stream(
    system: ResourceBoundedIncrementalInduction,
    character_indices: list[int],
    character_vocabulary: CharacterVocabulary,
    progress_description: str,
    scenario_name: str,
    clearml_logger: Any | None,
    program_snapshot_directory: Path | None,
    program_snapshot_interval_steps: int = 100,
    pool_snapshot_interval_steps: int = 250,
) -> ScenarioRunResult:
  uniform_loss_bits_per_character = math.log2(float(character_vocabulary.size))
  per_step_algorithm_loss_bits: list[float] = []
  per_step_baseline_loss_bits: list[float] = []
  previous_frozen_store_identifiers: tuple[str, ...] | None = None
  snapshot_interval_steps = max(1, int(pool_snapshot_interval_steps))
  program_snapshot_every = max(1, int(program_snapshot_interval_steps))

  with tqdm(total=len(character_indices),
            desc=progress_description,
            unit="ch",
            dynamic_ncols=True) as progress_bar:
    for step_index, observed_character_index in enumerate(character_indices,
                                                          start=1):
      system.step(observed_character_index=int(observed_character_index))

      loss_bits = system.last_loss_bits
      if loss_bits is None:
        loss_bits = uniform_loss_bits_per_character
      per_step_algorithm_loss_bits.append(float(loss_bits))
      per_step_baseline_loss_bits.append(float(uniform_loss_bits_per_character))
      frozen_store_identifiers = tuple(
        sorted(system.frozen_store.list_program_identifiers()))

      progress_bar.set_postfix({
        "loss_bits": f"{float(loss_bits):.3f}",
        "frozen_store": len(frozen_store_identifiers),
      }, refresh=False)
      progress_bar.update(1)

      if program_snapshot_directory is not None and (
          step_index % program_snapshot_every == 0):
        _write_program_snapshot_file(
          snapshot_directory=program_snapshot_directory,
          step_index=step_index,
          scenario_name=scenario_name,
          system=system,
        )

      if clearml_logger is None:
        continue

      clearml_logger.report_scalar(
        title="loss_bits",
        series=f"{scenario_name}/algorithm",
        value=float(loss_bits),
        iteration=step_index,
      )
      clearml_logger.report_scalar(
        title="loss_bits",
        series=f"{scenario_name}/baseline_uniform",
        value=float(uniform_loss_bits_per_character),
        iteration=step_index,
      )

      if (
          step_index == 1
          or step_index % snapshot_interval_steps == 0
          or step_index == len(character_indices)
      ):
        active_pool_snapshot = _format_active_pool_snapshot(
          system=system,
          step_index=step_index,
          scenario_name=scenario_name,
        )
        _report_text_channel(
          clearml_logger=clearml_logger,
          title="prediction_pool_snapshot",
          series=scenario_name,
          step=step_index,
          text_payload=active_pool_snapshot,
        )

      if frozen_store_identifiers != previous_frozen_store_identifiers:
        frozen_store_snapshot = _format_frozen_store_snapshot(
          system=system,
          step_index=step_index,
          scenario_name=scenario_name,
          frozen_program_identifiers=frozen_store_identifiers,
        )
        _report_text_channel(
          clearml_logger=clearml_logger,
          title="frozen_store_snapshot",
          series=scenario_name,
          step=step_index,
          text_payload=frozen_store_snapshot,
        )
        previous_frozen_store_identifiers = frozen_store_identifiers

  return ScenarioRunResult(
    per_step_algorithm_loss_bits=per_step_algorithm_loss_bits,
    per_step_baseline_loss_bits=per_step_baseline_loss_bits,
  )


def run_scenario(scenario, outputs_directory: Path,
                 clearml_logger: Any | None) -> None:
  vocabulary = CharacterVocabulary.from_text(scenario.stream_text)
  character_indices = vocabulary.encode_text(scenario.stream_text)

  configuration = ResourceBoundedIncrementalInductionConfiguration(
    pool_capacity=64,
    exploration_transformer_executions_per_step=32,
    validation_window_length=256,
    candidate_buffer_capacity=32,
    candidate_validation_interval=32,
    freeze_evaluation_interval=32,
    probability_floor=1e-3,
    detectability_slack_bits=8.0,
    reacquisition_tolerance_bits_per_character=0.25,
    maximum_recalled_programs_per_step=8,
    maximum_worker_threads=None,
  )

  primitive_library = create_default_primitive_library()
  memory_mechanism = CharacterHistoryMemoryMechanism(
    maximum_history_length=2048,
    maximum_context_length=16,
    recall_key_length=4,
    # smoothing_alpha=0.5,
    maximum_program_identifiers_per_key=8,
  )
  # transformers = [
  #   UniformPredictorTransformerProgram(),
  #   BigramPredictorTransformerProgram(),
  #   TrigramPredictorTransformerProgram(),
  #   RecallFrozenProgramTransformerProgram(),
  #   MixtureOfRecalledProgramsTransformerProgram(),
  #   DigitCycleStepEditTransformerProgram(step_change=1),
  #   DigitCycleStepEditTransformerProgram(step_change=-1),
  # ]
  # for step_size in range(1, 10):
  #   transformers.append(
  #     DigitCyclePredictorTransformerProgram(step_size=step_size))

  system = ResourceBoundedIncrementalInduction(
    character_vocabulary=vocabulary,
    configuration=configuration,
    primitive_library=primitive_library,
    # transformer_programs=transformers,
    memory_mechanism=memory_mechanism,
    # random_seed=0,
  )
  run_result = run_scenario_stream(
    system=system,
    character_indices=character_indices,
    character_vocabulary=vocabulary,
    progress_description=scenario.scenario_name,
    scenario_name=scenario.scenario_name,
    clearml_logger=clearml_logger,
    program_snapshot_directory=outputs_directory / scenario.scenario_name / "program_snapshots",
    program_snapshot_interval_steps=100,
    pool_snapshot_interval_steps=250,
  )

  reference_loss_bits = compute_reference_loss_bits_for_scenario(scenario,
                                                                 vocabulary)
  cumulative_algorithm_loss = \
  compute_cumulative_sum(run_result.per_step_algorithm_loss_bits)[-1]
  cumulative_baseline_loss = \
  compute_cumulative_sum(run_result.per_step_baseline_loss_bits)[-1]
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
  (
        scenario_outputs_directory / f"{scenario.scenario_name}_metrics.json").write_text(
    json.dumps(metrics_payload, indent=2),
    encoding="utf-8",
  )


def main() -> None:
  repository_directory = Path(__file__).resolve().parent
  data_directory = repository_directory / "data"
  outputs_directory = repository_directory / "outputs"
  outputs_directory.mkdir(parents=True, exist_ok=True)
  clearml_task = None
  clearml_logger = None
  if Task is not None:
    clearml_task = Task.init(project_name="rbii",
                             task_name="rbii_char_demo",
                             reuse_last_task_id=False)
    clearml_logger = clearml_task.get_logger()
  else:
    print(
      "ClearML is not installed; skipping ClearML logging. Install with: pip install clearml")

  scenarios = [
    build_scenario_a_context_switching(data_directory=data_directory),
    build_scenario_b_compositional_curriculum(),
    build_scenario_c_rare_glimpses(data_directory=data_directory),
    build_scenario_d_non_recurrent_drift(),
  ]

  documentation_primitive_library = create_default_primitive_library()
  _write_dsl_documentation_file(
    primitive_library=documentation_primitive_library,
    output_file_path=outputs_directory / "dsl_reference.lisp",
  )

  try:
    for scenario in scenarios:
      print(
        f"Running {scenario.scenario_name} (length={len(scenario.stream_text)})...")
      run_scenario(scenario,
                   outputs_directory=outputs_directory,
                   clearml_logger=clearml_logger)
  finally:
    if clearml_task is not None:
      clearml_task.close()

  print(f"Done. Outputs written to: {outputs_directory}")


if __name__ == "__main__":
  main()
