"""Data loaders for respiratory illness screening datasets.

Two supported datasets — choose based on availability:

1. COSWARA (recommended, CC BY 4.0)
   ─────────────────────────────────
   GitHub: https://github.com/iiscleap/Coswara-Data
   License: Creative Commons Attribution 4.0 International
   Download:
       git clone https://github.com/iiscleap/Coswara-Data.git

   Structure after clone:
       <root>/
           <date>/
               <participant_id>/
                   metadata.json         # health status, symptoms
                   breathing-shallow.wav
                   breathing-deep.wav
                   cough-shallow.wav
                   cough-heavy.wav
                   vowel-a.wav / vowel-e.wav / vowel-o.wav
                   counting-normal.wav
                   counting-fast.wav

   Labels: metadata["health_status"] — "healthy" → 0,
           "COVID-19" | "respiratory_illness" → 1

2. ICBHI 2017 Respiratory Sound Database
   ───────────────────────────────────────
   URL: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
   License: free for academic use after registration
   Download: register at URL above, then use the provided download link.

   Structure after download:
       <root>/
           <patient_id>_<session>_<equipment>.wav
           <patient_id>_<session>_<equipment>.txt   # annotation: onset, offset, crackle, wheeze
           ICBHI_Challenge_train_test.txt            # patient-level split

   Labels: presence of crackle/wheeze annotations → 1 (respiratory pathology),
           none → 0 (healthy)

STATUS: STUB — CoswaraLoader skeleton provided; ICBHI is a TODO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.data.base_loader import AudioSample, BaseLoader

COSWARA_POSITIVE_STATUSES = {
    "positive_mild",
    "positive_moderate",
    "positive_asymp",
    "resp_illness_not_identified",
}

# Statuses to skip (ambiguous/in-progress)
COSWARA_SKIP_STATUSES = {"under_validation", "recovered_full"}


class CoswaraLoader(BaseLoader):
    """Load the Coswara dataset for respiratory illness screening.

    Args:
        tasks: Which audio tasks to include.  ``None`` = all.
               Options: "breathing-shallow", "breathing-deep",
               "cough-shallow", "cough-heavy", "vowel-a", "vowel-e",
               "vowel-o", "counting-normal", "counting-fast".
    """

    condition = "respiratory"

    AUDIO_FILES = [
        "breathing-shallow",
        "breathing-deep",
        "cough-shallow",
        "cough-heavy",
        "vowel-a",
        "vowel-e",
        "vowel-o",
        "counting-normal",
        "counting-fast",
    ]

    def __init__(self, tasks: Optional[list[str]] = None) -> None:
        self.tasks = tasks or self.AUDIO_FILES

    def load_samples(
        self,
        root: str | Path,
        split: Optional[str] = None,
    ) -> list[AudioSample]:
        """Walk a cloned Coswara-Data directory and return AudioSamples.

        Args:
            root: Path to the cloned Coswara-Data repository root.
            split: Ignored (no pre-defined splits in Coswara).
        """
        root = Path(root)
        samples: list[AudioSample] = []

        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir() or date_dir.name.startswith("."):
                continue
            for participant_dir in sorted(date_dir.iterdir()):
                if not participant_dir.is_dir():
                    continue
                meta_path = participant_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception:
                    continue

                status = meta.get("covid_status", meta.get("health_status", "unknown"))
                if status in COSWARA_SKIP_STATUSES:
                    continue
                label = 1 if status in COSWARA_POSITIVE_STATUSES else 0

                for task in self.tasks:
                    wav = participant_dir / f"{task}.wav"
                    if not wav.exists():
                        continue
                    samples.append(
                        AudioSample(
                            path=wav,
                            label=label,
                            subject_id=participant_dir.name,
                            task=task,
                            condition=self.condition,
                            metadata={
                                "health_status": status,
                                "date": date_dir.name,
                                "age": meta.get("a", None),
                                "gender": meta.get("g", None),
                            },
                        )
                    )

        return samples


class ICBHILoader(BaseLoader):
    """Load the ICBHI 2017 Respiratory Sound Database.

    STATUS: TODO — implement after downloading the dataset.
    """

    condition = "respiratory"

    def load_samples(
        self,
        root: str | Path,
        split: Optional[str] = None,
    ) -> list[AudioSample]:
        raise NotImplementedError(
            "ICBHI 2017 loader not yet implemented. "
            "Register at https://bhichallenge.med.auth.gr/ to download."
        )
