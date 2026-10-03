from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
TARGET = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official MediaPipe model to {TARGET}")
    urllib.request.urlretrieve(URL, TARGET)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
