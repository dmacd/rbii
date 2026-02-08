from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterVocabulary:
  characters: tuple[str, ...]
  character_to_index: dict[str, int]
  index_to_character: dict[int, str]

  @property
  def size(self) -> int:
    return len(self.characters)

  @staticmethod
  def from_text(text: str) -> "CharacterVocabulary":
    unique_characters = sorted(set(text))
    character_to_index = {character: index for index, character in
                          enumerate(unique_characters)}
    index_to_character = {index: character for character, index in
                          character_to_index.items()}
    return CharacterVocabulary(
      characters=tuple(unique_characters),
      character_to_index=character_to_index,
      index_to_character=index_to_character,
    )

  def encode_text(self, text: str) -> list[int]:
    return [self.character_to_index[character] for character in text]

  def decode_indices(self, indices: list[int]) -> str:
    return "".join(self.index_to_character[index] for index in indices)
