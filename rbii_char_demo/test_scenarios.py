from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corpora import (
  build_paired_word_text,
  generate_numeric_progression_text,
  generate_random_markov_text,
  load_dr_seuss_text,
  load_shakespeare_text,
)


@dataclass(frozen=True)
class EpisodeDescription:
  start_index: int
  end_index: int
  task_name: str


@dataclass(frozen=True)
class ScenarioDescription:
  scenario_name: str
  stream_text: str
  episodes: list[EpisodeDescription]


def _append_episode(
    episodes: list[EpisodeDescription],
    current_text_parts: list[str],
    task_name: str,
    episode_text: str,
) -> None:
  start_index = sum(len(part) for part in current_text_parts)
  current_text_parts.append(episode_text)
  end_index = start_index + len(episode_text)
  episodes.append(
    EpisodeDescription(start_index=start_index, end_index=end_index,
                       task_name=task_name))


def build_scenario_a_context_switching(
    data_directory: Path) -> ScenarioDescription:
  shakespeare_text = load_shakespeare_text(repetitions=50)
  dr_seuss_text = load_dr_seuss_text(data_directory=data_directory,
                                     repetitions=120)
  paired_word_text = build_paired_word_text(repetitions=120)

  episode_length = 3000
  cycle_count = 2

  episodes: list[EpisodeDescription] = []
  stream_text_parts: list[str] = []

  shakespeare_offset = 0
  dr_seuss_offset = 0
  paired_word_offset = 0

  for cycle_index in range(cycle_count):
    episode_text = shakespeare_text[
      shakespeare_offset: shakespeare_offset + episode_length]
    shakespeare_offset = (shakespeare_offset + episode_length) % max(1, len(
      shakespeare_text) - episode_length)
    _append_episode(episodes, stream_text_parts, "shakespeare", episode_text)

    episode_text = dr_seuss_text[
      dr_seuss_offset: dr_seuss_offset + episode_length]
    dr_seuss_offset = (dr_seuss_offset + episode_length) % max(1, len(
      dr_seuss_text) - episode_length)
    _append_episode(episodes, stream_text_parts, "dr_seuss", episode_text)

    episode_text = generate_numeric_progression_text(step_size=1,
                                                     length=episode_length)
    _append_episode(episodes, stream_text_parts, "numeric_progression_step_1",
                    episode_text)

    episode_text = paired_word_text[
      paired_word_offset: paired_word_offset + episode_length]
    paired_word_offset = (paired_word_offset + episode_length) % max(1, len(
      paired_word_text) - episode_length)
    _append_episode(episodes, stream_text_parts, "paired_word", episode_text)

  return ScenarioDescription(
    scenario_name="scenario_a_context_switching",
    stream_text="".join(stream_text_parts),
    episodes=episodes,
  )


def build_scenario_b_compositional_curriculum() -> ScenarioDescription:
  episode_length = 3000
  step_sizes = [1, 2, 3, 4, 5, 1, 2]

  episodes: list[EpisodeDescription] = []
  stream_text_parts: list[str] = []

  for step_size in step_sizes:
    task_name = f"numeric_progression_step_{step_size}"
    episode_text = generate_numeric_progression_text(step_size=step_size,
                                                     length=episode_length)
    _append_episode(episodes, stream_text_parts, task_name, episode_text)

  return ScenarioDescription(
    scenario_name="scenario_b_compositional_curriculum",
    stream_text="".join(stream_text_parts),
    episodes=episodes,
  )


def build_scenario_c_rare_glimpses(data_directory: Path) -> ScenarioDescription:
  shakespeare_text = load_shakespeare_text(repetitions=120)
  paired_word_text = build_paired_word_text(repetitions=40)

  long_text_length = 16000
  glimpse_length = 120
  glimpse_count = 8

  base_text = shakespeare_text[:long_text_length]

  episodes: list[EpisodeDescription] = []
  stream_text_parts: list[str] = []

  # Interleave: long Shakespeare chunks punctuated by short paired-word glimpses.
  chunk_length = long_text_length // (glimpse_count + 1)
  paired_word_offset = 0
  for index in range(glimpse_count + 1):
    start = index * chunk_length
    end = min(len(base_text), (index + 1) * chunk_length)
    _append_episode(episodes, stream_text_parts, "shakespeare_background",
                    base_text[start:end])

    if index < glimpse_count:
      glimpse = paired_word_text[
        paired_word_offset: paired_word_offset + glimpse_length]
      paired_word_offset = (paired_word_offset + glimpse_length) % max(1, len(
        paired_word_text) - glimpse_length)
      _append_episode(episodes, stream_text_parts, "paired_word_glimpse",
                      glimpse)

  return ScenarioDescription(
    scenario_name="scenario_c_rare_glimpses",
    stream_text="".join(stream_text_parts),
    episodes=episodes,
  )


def build_scenario_d_non_recurrent_drift() -> ScenarioDescription:
  # Random Markov episodes that never repeat.
  episode_length = 4000
  episode_count = 6
  alphabet = "abcdefghijklmnopqrstuvwxyz .,;\n"

  episodes: list[EpisodeDescription] = []
  stream_text_parts: list[str] = []

  for episode_index in range(episode_count):
    task_name = f"random_drift_{episode_index}"
    episode_text = generate_random_markov_text(
      alphabet=alphabet,
      length=episode_length,
      random_seed=episode_index,
    )
    _append_episode(episodes, stream_text_parts, task_name, episode_text)

  return ScenarioDescription(
    scenario_name="scenario_d_non_recurrent_drift",
    stream_text="".join(stream_text_parts),
    episodes=episodes,
  )
