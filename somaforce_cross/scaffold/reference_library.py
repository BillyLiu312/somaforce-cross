"""Storage and selection helpers for canonical scaffold references."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from somaforce_cross.scaffold.contracts import ScaffoldTask
from somaforce_cross.scaffold.reference_schema import (
    CanonicalReferenceEpisode,
    ReferenceSource,
)


class ReferenceLibrary:
    """In-memory canonical reference library with deterministic persistence."""

    def __init__(self, episodes: Iterable[CanonicalReferenceEpisode] = ()) -> None:
        self._episodes: dict[str, CanonicalReferenceEpisode] = {}
        for episode in episodes:
            self.add(episode)

    def add(self, episode: CanonicalReferenceEpisode) -> None:
        key = episode.metadata.episode_id
        if key in self._episodes:
            raise ValueError(f"Duplicate canonical reference episode_id: {key}")
        self._episodes[key] = episode

    def get(self, episode_id: str) -> CanonicalReferenceEpisode:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown canonical reference episode_id: {episode_id}"
            ) from exc

    def select(
        self,
        *,
        task: ScaffoldTask | None = None,
        source: ReferenceSource | None = None,
        valid_only: bool = True,
    ) -> list[CanonicalReferenceEpisode]:
        episodes = []
        for episode in self._episodes.values():
            if task is not None and episode.metadata.task != task:
                continue
            if source is not None and episode.metadata.source != source:
                continue
            if valid_only and not bool(torch.any(episode.reference_valid)):
                continue
            episodes.append(episode)
        return episodes

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "library_schema_version": 1,
            "episodes": [
                episode.as_serializable() for episode in self._episodes.values()
            ],
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls, path: Path, map_location: str | torch.device = "cpu"
    ) -> ReferenceLibrary:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if payload.get("library_schema_version") != 1:
            raise ValueError("Unsupported reference library schema version")
        return cls(
            CanonicalReferenceEpisode.from_serializable(raw)
            for raw in payload["episodes"]
        )

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes.values())
