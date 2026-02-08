from __future__ import annotations

from pathlib import Path
import random


def load_shakespeare_text(repetitions: int = 40) -> str:
  # Public domain excerpt (Sonnet 18).
  excerpt = (
    "Shall I compare thee to a summer's day?\n"
    "Thou art more lovely and more temperate:\n"
    "Rough winds do shake the darling buds of May,\n"
    "And summer's lease hath all too short a date;\n"
    "Sometime too hot the eye of heaven shines,\n"
    "And often is his gold complexion dimm'd;\n"
    "And every fair from fair sometime declines,\n"
    "By chance, or nature's changing course, untrimm'd;\n"
    "But thy eternal summer shall not fade,\n"
    "Nor lose possession of that fair thou ow'st;\n"
    "Nor shall death brag thou wander'st in his shade,\n"
    "When in eternal lines to time thou grow'st:\n"
    "So long as men can breathe, or eyes can see,\n"
    "So long lives this, and this gives life to thee.\n"
  )
  return excerpt * int(repetitions)


def load_dr_seuss_text(data_directory: Path, repetitions: int = 80) -> str:
  # Dr. Seuss works are copyrighted; this demo does not bundle them.
  # If you have a legal corpus, place it at data/dr_seuss.txt
  file_path = data_directory / "dr_seuss.txt"
  if file_path.exists():
    return file_path.read_text(encoding="utf-8", errors="ignore")

  synthetic_placeholder = (
    "I like to hop. I like to bop.\n"
    "I like to flip, then stop, then flop.\n"
    "A fish can wish, a dish can swish.\n"
    "We rhyme in time with lines that chime.\n"
    "Up we go, down we go, round we go again.\n"
  )
  return synthetic_placeholder * int(repetitions)


def build_paired_word_text(repetitions: int = 200) -> str:
  # A tiny built-in thesaurus-like list (safe to ship).
  synonym_pairs = [
    ("big", "large"),
    ("small", "tiny"),
    ("happy", "joyful"),
    ("sad", "unhappy"),
    ("fast", "quick"),
    ("slow", "sluggish"),
    ("smart", "clever"),
    ("angry", "irate"),
    ("calm", "serene"),
    ("bright", "luminous"),
  ]
  lines = []
  for _ in range(int(repetitions)):
    for left_word, right_word in synonym_pairs:
      lines.append(f"{left_word}->{right_word}\n")
  return "".join(lines)


def generate_numeric_progression_text(step_size: int, length: int) -> str:
  step = int(step_size) % 10
  digits = []
  value = 0
  for _ in range(int(length)):
    digits.append(str(value))
    value = (value + step) % 10
  return "".join(digits)


def generate_random_markov_text(
    alphabet: str,
    length: int,
    random_seed: int,
) -> str:
  random_generator = random.Random(int(random_seed))
  vocabulary_size = len(alphabet)
  # Random transition matrix
  transition_probabilities = []
  for _ in range(vocabulary_size):
    raw = [random_generator.random() + 1e-6 for _ in range(vocabulary_size)]
    total = sum(raw)
    transition_probabilities.append([value / total for value in raw])

  indices = []
  current_index = random_generator.randrange(vocabulary_size)
  indices.append(current_index)
  for _ in range(int(length) - 1):
    probabilities = transition_probabilities[current_index]
    r = random_generator.random()
    cumulative = 0.0
    next_index = 0
    for index, probability in enumerate(probabilities):
      cumulative += probability
      if r <= cumulative:
        next_index = index
        break
    current_index = next_index
    indices.append(current_index)

  return "".join(alphabet[index] for index in indices)
