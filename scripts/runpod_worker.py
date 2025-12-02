#!/usr/bin/env python3
"""
RunPod worker that wraps the XTTS + LivePortrait talking-head pipeline.

Input event schema:
{
    "image_path": "/workspace/input.jpg",          # optional if image_url is provided
    "image_url": "https://....jpg",                # downloaded to a temp file if given
    "text_prompt": "Hello world",                  # required unless text_file_url is set
    "text_file_url": "https://.../script.txt",     # optional: fetch text from URL
    "speaker_wav_url": "https://.../voice.wav",    # optional voice clone reference
    "speaker_wav_path": "/workspace/voice.wav",    # optional local voice clone reference
    "output_dir": "/workspace/outputs",            # optional, defaults to /workspace/outputs
    "output_name": "final_talking_video.mp4",      # optional filename
    "language": "en",                              # XTTS language
    "fps": 25,                                     # LivePortrait fps
    "use_gpu": true,                               # enable CUDA if available
    "liveportrait_entry": "/workspace/LivePortrait/inference.py",  # path to entrypoint
    "keep_intermediate": false,                    # whether to keep speech.wav and silent.mp4
    "liveportrait_extra": ["--some-flag"],         # passthrough args to LivePortrait
    "return_base64": false                         # if true, returns base64 of final video
}
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import runpod
import requests

from scripts.generate_talking_videos import generate_talking_video


def _download(url: str, suffix: str) -> Path:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    path = Path(tmp_path)
    path.write_bytes(resp.content)
    return path


def _resolve_file(local_path: Optional[str], url: Optional[str], suffix: str) -> Path:
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {local_path}")
        return path
    if url:
        return _download(url, suffix)
    raise ValueError(f"Expected a local path or URL for {suffix}")


def _resolve_text(text: Optional[str], text_file_url: Optional[str]) -> str:
    if text:
        return text
    if text_file_url:
        return requests.get(text_file_url, timeout=60).text
    raise ValueError("text_prompt or text_file_url is required")


def handler(event: Dict) -> Dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # RunPod serverless passes user data under event["input"].
    payload = event.get("input", event)

    image_path = _resolve_file(payload.get("image_path"), payload.get("image_url"), suffix=".jpg")
    text_prompt = _resolve_text(payload.get("text_prompt"), payload.get("text_file_url"))

    speaker_wav = None
    if payload.get("speaker_wav_path") or payload.get("speaker_wav_url"):
        speaker_wav = _resolve_file(
            payload.get("speaker_wav_path"),
            payload.get("speaker_wav_url"),
            suffix=".wav",
        )

    output_dir = Path(payload.get("output_dir", "/workspace/outputs"))
    output_name = payload.get("output_name", "final_talking_video.mp4")

    final_video = generate_talking_video(
        image_path=image_path,
        text_prompt=text_prompt,
        output_dir=output_dir,
        speaker_wav=speaker_wav,
        speaker_id=payload.get("speaker_id"),
        language=payload.get("language", "en"),
        fps=int(payload.get("fps", 25)),
        liveportrait_entry=Path(payload["liveportrait_entry"]) if payload.get("liveportrait_entry") else None,
        use_gpu=bool(payload.get("use_gpu", True)),
        keep_intermediate=bool(payload.get("keep_intermediate", False)),
        extra_liveportrait_args=payload.get("liveportrait_extra"),
        output_name=output_name,
    )

    return_payload = {
        "final_video_path": str(final_video),
    }

    if payload.get("return_base64"):
        video_bytes = final_video.read_bytes()
        return_payload["final_video_base64"] = base64.b64encode(video_bytes).decode("utf-8")

    return return_payload


runpod.serverless.start({"handler": handler})
