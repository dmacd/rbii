from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as matplotlib_plotting  # noqa: E402

from metrics import (
  compute_cumulative_sum,
  compute_cumulative_compression_gain_bits,
  ReacquisitionMeasurement,
)


def plot_cumulative_loss(
    output_file_path: Path,
    per_step_algorithm_loss_bits: list[float],
    per_step_baseline_loss_bits: list[float],
    per_step_reference_loss_bits: list[float] | None = None,
    title: str = "Cumulative loss (bits)",
) -> None:
  output_file_path.parent.mkdir(parents=True, exist_ok=True)

  time_steps = list(range(1, len(per_step_algorithm_loss_bits) + 1))
  algorithm_cumulative = compute_cumulative_sum(per_step_algorithm_loss_bits)
  baseline_cumulative = compute_cumulative_sum(per_step_baseline_loss_bits)

  figure = matplotlib_plotting.figure()
  matplotlib_plotting.plot(time_steps, algorithm_cumulative, label="rbii_demo")
  matplotlib_plotting.plot(time_steps, baseline_cumulative,
                           label="uniform_baseline")

  if per_step_reference_loss_bits is not None:
    reference_cumulative = compute_cumulative_sum(per_step_reference_loss_bits)
    matplotlib_plotting.plot(time_steps, reference_cumulative,
                             label="reference_bigram")

  matplotlib_plotting.xlabel("time step")
  matplotlib_plotting.ylabel("cumulative loss (bits)")
  matplotlib_plotting.title(title)
  matplotlib_plotting.legend()
  figure.tight_layout()
  figure.savefig(output_file_path)
  matplotlib_plotting.close(figure)


def plot_cumulative_compression_gain(
    output_file_path: Path,
    per_step_algorithm_loss_bits: list[float],
    per_step_baseline_loss_bits: list[float],
    title: str = "Cumulative compression gain (bits saved vs uniform)",
) -> None:
  output_file_path.parent.mkdir(parents=True, exist_ok=True)
  time_steps = list(range(1, len(per_step_algorithm_loss_bits) + 1))
  cumulative_gain = compute_cumulative_compression_gain_bits(
    per_step_algorithm_loss_bits=per_step_algorithm_loss_bits,
    per_step_baseline_loss_bits=per_step_baseline_loss_bits,
  )

  figure = matplotlib_plotting.figure()
  matplotlib_plotting.plot(time_steps, cumulative_gain, label="rbii_demo_gain")
  matplotlib_plotting.xlabel("time step")
  matplotlib_plotting.ylabel("cumulative gain (bits)")
  matplotlib_plotting.title(title)
  matplotlib_plotting.legend()
  figure.tight_layout()
  figure.savefig(output_file_path)
  matplotlib_plotting.close(figure)


def plot_reacquisition_delay(
    output_file_path: Path,
    measurements: list[ReacquisitionMeasurement],
    title: str = "Reacquisition delay (steps until low regret)",
) -> None:
  output_file_path.parent.mkdir(parents=True, exist_ok=True)
  figure = matplotlib_plotting.figure()

  task_names = sorted(
    set(measurement.task_name for measurement in measurements))
  for task_name in task_names:
    task_measurements = [m for m in measurements if m.task_name == task_name]
    return_indices = [m.return_index for m in task_measurements]
    delays = [m.reacquisition_delay for m in task_measurements]
    matplotlib_plotting.plot(return_indices, delays, marker="o",
                             label=task_name)

  matplotlib_plotting.xlabel("return index (episode number for the task)")
  matplotlib_plotting.ylabel("reacquisition delay (steps)")
  matplotlib_plotting.title(title)
  matplotlib_plotting.legend()
  figure.tight_layout()
  figure.savefig(output_file_path)
  matplotlib_plotting.close(figure)


def plot_reacquisition_excess_loss(
    output_file_path: Path,
    measurements: list[ReacquisitionMeasurement],
    title: str = "Reacquisition excess loss (regret paid before reacquisition)",
) -> None:
  output_file_path.parent.mkdir(parents=True, exist_ok=True)
  figure = matplotlib_plotting.figure()

  task_names = sorted(
    set(measurement.task_name for measurement in measurements))
  for task_name in task_names:
    task_measurements = [m for m in measurements if m.task_name == task_name]
    return_indices = [m.return_index for m in task_measurements]
    excess_losses = [m.reacquisition_excess_loss_bits for m in
                     task_measurements]
    matplotlib_plotting.plot(return_indices, excess_losses, marker="o",
                             label=task_name)

  matplotlib_plotting.xlabel("return index (episode number for the task)")
  matplotlib_plotting.ylabel("excess loss (bits)")
  matplotlib_plotting.title(title)
  matplotlib_plotting.legend()
  figure.tight_layout()
  figure.savefig(output_file_path)
  matplotlib_plotting.close(figure)
