"""Abstract base class for all condition-specific data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioSample:
    """Unified audio sample representation across all conditions."""

    path: Path
    label: int         # 0 = control / negative, 1 = condition positive
    subject_id: str
    task: str          # recording task / prompt type
    condition: str     # "parkinson" | "depression" | "respiratory"
    metadata: dict     # condition-specific extras (age, severity, etc.)


class BaseLoader(ABC):
    """Defines the interface every condition loader must implement."""

    condition: str  # override in subclass

    @abstractmethod
    def load_samples(
        self,
        root: str | Path,
        split: Optional[str] = None,
    ) -> list[AudioSample]:
        """Return all audio samples from *root* for the given *split*.

        Args:
            root: Path to the dataset root directory.
            split: Optional "train" | "val" | "test". If None, return all.

        Returns:
            Flat list of AudioSample dataclass instances.
        """

    def subject_ids(self, samples: list[AudioSample]) -> list[str]:
        """Unique subject identifiers, preserving insertion order."""
        seen: dict[str, None] = {}
        for s in samples:
            seen[s.subject_id] = None
        return list(seen)
