"""Data loader for the Italian Parkinson's Voice and Speech corpus.

Directory layout expected:
    <root>/
      15 Young Healthy Control/
        <subject_name>/
          <task><code>.wav
      22 Elderly Healthy Control/
        <subject_name>/
          *.wav
      28 People with Parkinson's disease/
        <updrs_range>/      # e.g. "1-5", "6-10", "11-16", "17-28"
          <subject_name>/
            *.wav

Task code prefixes found in filenames:
  B1, B2      — sustained vowel /a/ (two sessions)
  VA,VE,VI,VO,VU — sustained Italian vowels /a/ /e/ /i/ /o/ /u/
  D1          — diadochokinetic task (/pa/ /ta/ /ka/ repetitions)
  PR          — passage reading
  FB          — free/spontaneous speech

Labels:  0 = healthy control,  1 = Parkinson's disease
UPDRS range is stored as metadata; use it to stratify severity analyses.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.data.base_loader import AudioSample, BaseLoader

_UPDRS_RE = re.compile(r"^\d+-\d+$")
_TASK_RE  = re.compile(r"^([A-Z]+\d*)")

LABEL_PD = 1
LABEL_HC = 0

TASK_VOWELS      = {"B1", "B2", "VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"}
TASK_KINETIC     = {"D1", "D2"}
TASK_READING     = {"PR1", "PR11"}
TASK_SPONTANEOUS = {"FB1"}
TASK_ALL         = TASK_VOWELS | TASK_KINETIC | TASK_READING | TASK_SPONTANEOUS


def _parse_task(stem: str) -> str:
    m = _TASK_RE.match(stem)
    return m.group(1) if m else "UNKNOWN"


class ParkinsonLoader(BaseLoader):
    """Load the Italian Parkinson's Voice and Speech corpus.

    Args:
        tasks: Task filter. Pass a set of task codes to restrict which
               recordings are loaded, e.g. ``{"B1", "B2"}`` for sustained
               vowel tasks only. ``None`` loads everything.
        include_young_controls: Whether to include the 15 young healthy
               controls (different age range from PD patients). Default True.
    """

    condition = "parkinson"

    def __init__(
        self,
        tasks: Optional[set[str]] = None,
        include_young_controls: bool = True,
    ) -> None:
        self.tasks = tasks
        self.include_young_controls = include_young_controls

    # ------------------------------------------------------------------

    def load_samples(
        self,
        root: str | Path,
        split: Optional[str] = None,
    ) -> list[AudioSample]:
        """Walk *root* and return all matching AudioSample objects.

        ``split`` is accepted for :class:`BaseLoader` API compatibility but
        unused here — call :func:`subject_split` to divide by subject ID.
        """
        del split  # intentionally unused; subject split is done externally
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        samples: list[AudioSample] = []

        for group_dir in sorted(root.iterdir()):
            if not group_dir.is_dir():
                continue

            name = group_dir.name
            if "Young Healthy" in name:
                if not self.include_young_controls:
                    continue
                label, age_group = LABEL_HC, "young_healthy"
                samples.extend(self._walk_subjects(group_dir, label, age_group, ""))
            elif "Elderly Healthy" in name:
                label, age_group = LABEL_HC, "elderly_healthy"
                samples.extend(self._walk_subjects(group_dir, label, age_group, ""))
            elif "Parkinson" in name:
                samples.extend(self._walk_pd(group_dir))

        return samples

    # ------------------------------------------------------------------

    def _walk_subjects(
        self,
        group_dir: Path,
        label: int,
        age_group: str,
        updrs_range: str,
    ) -> list[AudioSample]:
        samples: list[AudioSample] = []
        for subj_dir in sorted(group_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for wav in sorted(subj_dir.glob("*.wav")):
                task = _parse_task(wav.stem)
                if self.tasks and task not in self.tasks:
                    continue
                samples.append(
                    AudioSample(
                        path=wav,
                        label=label,
                        subject_id=subj_dir.name,
                        task=task,
                        condition=self.condition,
                        metadata={
                            "age_group": age_group,
                            "updrs_range": updrs_range,
                        },
                    )
                )
        return samples

    def _walk_pd(self, pd_root: Path) -> list[AudioSample]:
        """Walk the Parkinson's group directory.

        Expected layout: pd_root/{updrs_range}/{subject_name}/*.wav
        """
        samples: list[AudioSample] = []
        for child in sorted(pd_root.iterdir()):
            if not child.is_dir():
                continue
            if _UPDRS_RE.match(child.name):
                # severity subdirectory → subjects inside
                samples.extend(
                    self._walk_subjects_flat(child, LABEL_PD, "parkinson", child.name)
                )
        return samples

    def _walk_subjects_flat(
        self,
        severity_dir: Path,
        label: int,
        age_group: str,
        updrs_range: str,
    ) -> list[AudioSample]:
        """Walk severity_dir/{subject_name}/*.wav."""
        samples: list[AudioSample] = []
        for subj_dir in sorted(severity_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for wav in sorted(subj_dir.glob("*.wav")):
                task = _parse_task(wav.stem)
                if self.tasks and task not in self.tasks:
                    continue
                samples.append(
                    AudioSample(
                        path=wav,
                        label=label,
                        subject_id=subj_dir.name,
                        task=task,
                        condition=self.condition,
                        metadata={
                            "age_group": age_group,
                            "updrs_range": updrs_range,
                        },
                    )
                )
        return samples


# ------------------------------------------------------------------
# Subject-level train/val/test split helpers
# ------------------------------------------------------------------

def subject_split(
    samples: list[AudioSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[AudioSample], list[AudioSample], list[AudioSample]]:
    """Split *samples* by subject so no subject appears in multiple sets.

    Stratifies by label to maintain class balance across splits.
    Returns (train, val, test) lists.
    """
    import random

    rng = random.Random(seed)

    # Group subject IDs by label
    by_label: dict[int, list[str]] = {}
    subject_to_label: dict[str, int] = {}
    for s in samples:
        subject_to_label[s.subject_id] = s.label
        by_label.setdefault(s.label, [])
        if s.subject_id not in by_label[s.label]:
            by_label[s.label].append(s.subject_id)

    train_ids, val_ids, test_ids = set(), set(), set()

    for _, ids in by_label.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_ratio))
        n_val  = max(1, round(len(ids) * val_ratio))
        test_ids.update(ids[:n_test])
        val_ids.update(ids[n_test: n_test + n_val])
        train_ids.update(ids[n_test + n_val:])

    train = [s for s in samples if s.subject_id in train_ids]
    val   = [s for s in samples if s.subject_id in val_ids]
    test  = [s for s in samples if s.subject_id in test_ids]
    return train, val, test
