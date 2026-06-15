"""Data loader for depression affect screening.

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)
License: CC BY-NC-SA 4.0
Download: https://zenodo.org/record/1188976
          → Audio_Speech_Actors_01-24.zip (198 MB)

Labelling strategy (research proxy — NOT clinical depression):
  Depressed affect (label=1): sad (04), fearful (06)
  Neutral affect  (label=0): neutral (01), calm (02), happy (03)
  Skipped:                   angry (05), disgust (07), surprised (08)

RAVDESS filename convention:
  03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
  Emotion codes: 01=neutral 02=calm 03=happy 04=sad
                 05=angry   06=fearful 07=disgust 08=surprised
  Intensity: 01=normal 02=strong
  Actor: 01-24  (odd=male, even=female)

Structure after unzip:
  <root>/
    Actor_01/
      03-01-01-01-01-01-01.wav
      ...
    Actor_24/
      ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.data.base_loader import AudioSample, BaseLoader

DEPRESSED_EMOTIONS = {"04", "06"}    # sad, fearful
NEUTRAL_EMOTIONS   = {"01", "02", "03"}  # neutral, calm, happy


class RAVDESSLoader(BaseLoader):
    """Load RAVDESS speech data as a depression-affect proxy.

    Args:
        emotions: Override which emotion codes to include (None = use defaults).
    """

    condition = "depression"

    def __init__(self, emotions: Optional[set[str]] = None) -> None:
        self.depressed = emotions or DEPRESSED_EMOTIONS
        self.neutral   = NEUTRAL_EMOTIONS

    def load_samples(
        self,
        root: str | Path,
        split: Optional[str] = None,
    ) -> list[AudioSample]:
        """Walk the unzipped RAVDESS directory and return AudioSamples.

        Args:
            root: Path containing Actor_01/ … Actor_24/.
            split: Ignored.
        """
        root = Path(root)
        samples: list[AudioSample] = []

        for actor_dir in sorted(root.iterdir()):
            if not actor_dir.is_dir() or not actor_dir.name.startswith("Actor_"):
                continue
            actor_id = actor_dir.name

            for wav in sorted(actor_dir.glob("*.wav")):
                parts = wav.stem.split("-")
                if len(parts) < 7:
                    continue
                emotion_code = parts[2]

                if emotion_code in self.depressed:
                    label = 1
                elif emotion_code in self.neutral:
                    label = 0
                else:
                    continue

                samples.append(
                    AudioSample(
                        path=wav,
                        label=label,
                        subject_id=actor_id,
                        task="speech",
                        condition=self.condition,
                        metadata={
                            "emotion_code": emotion_code,
                            "intensity": parts[3],
                            "statement": parts[4],
                            "gender": "male" if int(parts[6]) % 2 == 1 else "female",
                        },
                    )
                )

        return samples
