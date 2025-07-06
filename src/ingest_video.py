"""Video ingestion: transcribe with whisper.cpp & grab frames."""
from pathlib import Path
from typing import List

def ingest_video(url_or_path: str, output_dir: Path, frame_rate: int = 2) -> List[Path]:
    """Downloads (if needed), transcribes, extracts frames; returns metadata."""
    # TODO: yt-dlp + ffmpeg + whisper-cpp
    raise NotImplementedError
