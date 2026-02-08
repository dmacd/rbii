from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as numpy

from character_vocabulary import CharacterVocabulary
from test_scenarios import ScenarioDescription, EpisodeDescription


@dataclass(frozen=True)
class ReacquisitionMeasurement:
  task_name: str
  return_index: int
  episode_start_index: int
  episode_end_index: int
  episode_length: int
  reacquisition_delay: int
  reacquisition_excess_loss_bits: float


class CharacterBigramReferenceModel:
  def __init__(self, character_vocabulary: CharacterVocabulary,
               training_indices: list[int],
               smoothing_alpha: float = 0.5) -> None:
    self.character_vocabulary = character_vocabulary
    self.smoothing_alpha = float(smoothing_alpha)
    vocabulary_size = character_vocabulary.size
    self.bigram_counts = numpy.zeros((vocabulary_size, vocabulary_size),
                                     dtype=numpy.int64)
    self.unigram_counts = numpy.zeros(vocabulary_size, dtype=numpy.int64)

    if len(training_indices) >= 1:
      self.unigram_counts[training_indices[0]] += 1
    for previous_index, next_index in zip(training_indices[:-1],
                                          training_indices[1:]):
      self.bigram_counts[int(previous_index), int(next_index)] += 1
      self.unigram_counts[int(next_index)] += 1

  def predict_distribution(self,
                           previous_character_index: int | None) -> numpy.ndarray:
    vocabulary_size = self.character_vocabulary.size
    if previous_character_index is None:
      counts = self.unigram_counts
    else:
      counts = self.bigram_counts[int(previous_character_index)]
    smoothed = counts.astype(numpy.float64) + self.smoothing_alpha
    total = float(smoothed.sum())
    if total <= 0.0:
      return numpy.ones(vocabulary_size, dtype=numpy.float64) / float(
        vocabulary_size)
    return smoothed / total


def compute_reference_loss_bits_for_scenario(
    scenario: ScenarioDescription,
    character_vocabulary: CharacterVocabulary,
) -> list[float]:
  # Train one bigram model per task name using all text belonging to that task in the scenario.
  task_name_to_training_text: dict[str, str] = {}
  for episode in scenario.episodes:
    episode_text = scenario.stream_text[episode.start_index: episode.end_index]
    task_name_to_training_text[
      episode.task_name] = task_name_to_training_text.get(episode.task_name,
                                                          "") + episode_text

  task_name_to_model: dict[str, CharacterBigramReferenceModel] = {}
  for task_name, training_text in task_name_to_training_text.items():
    training_indices = character_vocabulary.encode_text(training_text)
    task_name_to_model[task_name] = CharacterBigramReferenceModel(
      character_vocabulary, training_indices)

  # Compute reference loss aligned with the global stream.
  reference_loss_bits: list[float] = [0.0 for _ in
                                      range(len(scenario.stream_text))]
  for episode in scenario.episodes:
    model = task_name_to_model[episode.task_name]
    episode_indices = character_vocabulary.encode_text(
      scenario.stream_text[episode.start_index: episode.end_index])
    previous_index: int | None = None
    for offset, next_index in enumerate(episode_indices):
      distribution = model.predict_distribution(previous_index)
      probability_value = float(distribution[int(next_index)])
      reference_loss_bits[episode.start_index + offset] = -math.log2(
        probability_value)
      previous_index = int(next_index)

  return reference_loss_bits


def compute_cumulative_sum(values: list[float]) -> list[float]:
  cumulative = []
  running_total = 0.0
  for value in values:
    running_total += float(value)
    cumulative.append(running_total)
  return cumulative


def compute_cumulative_compression_gain_bits(
    per_step_algorithm_loss_bits: list[float],
    per_step_baseline_loss_bits: list[float],
) -> list[float]:
  if len(per_step_algorithm_loss_bits) != len(per_step_baseline_loss_bits):
    raise ValueError("Loss arrays must have the same length.")
  cumulative_gain = []
  running_total = 0.0
  for algorithm_loss, baseline_loss in zip(per_step_algorithm_loss_bits,
                                           per_step_baseline_loss_bits):
    running_total += float(baseline_loss) - float(algorithm_loss)
    cumulative_gain.append(running_total)
  return cumulative_gain


def compute_reacquisition_measurements(
    scenario: ScenarioDescription,
    per_step_algorithm_loss_bits: list[float],
    per_step_reference_loss_bits: list[float],
    tolerance_bits_per_character: float,
) -> list[ReacquisitionMeasurement]:
  if len(per_step_algorithm_loss_bits) != len(per_step_reference_loss_bits):
    raise ValueError(
      "Algorithm and reference loss arrays must have the same length.")

  task_name_to_episodes: dict[str, list[EpisodeDescription]] = {}
  for episode in scenario.episodes:
    task_name_to_episodes.setdefault(episode.task_name, []).append(episode)

  measurements: list[ReacquisitionMeasurement] = []
  for task_name, episodes in task_name_to_episodes.items():
    if len(episodes) < 2:
      continue

    for return_index, episode in enumerate(episodes):
      episode_algorithm_losses = per_step_algorithm_loss_bits[
        episode.start_index: episode.end_index]
      episode_reference_losses = per_step_reference_loss_bits[
        episode.start_index: episode.end_index]
      episode_length = len(episode_algorithm_losses)

      cumulative_regret = 0.0
      reacquisition_delay = episode_length
      reacquisition_excess_loss_bits = 0.0

      for prefix_length in range(1, episode_length + 1):
        cumulative_regret += float(
          episode_algorithm_losses[prefix_length - 1]) - float(
          episode_reference_losses[prefix_length - 1]
        )
        average_regret = cumulative_regret / float(prefix_length)
        if average_regret <= float(tolerance_bits_per_character):
          reacquisition_delay = prefix_length
          reacquisition_excess_loss_bits = cumulative_regret
          break

      measurements.append(
        ReacquisitionMeasurement(
          task_name=task_name,
          return_index=int(return_index),
          episode_start_index=int(episode.start_index),
          episode_end_index=int(episode.end_index),
          episode_length=int(episode_length),
          reacquisition_delay=int(reacquisition_delay),
          reacquisition_excess_loss_bits=float(reacquisition_excess_loss_bits),
        )
      )

  return measurements
