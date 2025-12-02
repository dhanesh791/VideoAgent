#!/usr/bin/env python3
"""
Alternative talking-head-style pipeline using XTTS + Wan 2.2 (experimental).

This is NOT the primary project pipeline (which must use LivePortrait).
It is provided separately because you explicitly requested a Wan 2.2 variant.

Intended flow (mirrors the LivePortrait version as much as possible):
    text → XTTS → audio.wav
    reference image + text (+ optional audio) → Wan 2.2 → video_no_audio.mp4
    ffmpeg merge (if Wan outputs silent video) → final_talking_video.mp4

Notes:
- This script assumes you have a local Wan 2.2 inference entrypoint and that
  you know its exact CLI/API. The `run_wan` function simply shells out to that
  entrypoint with generic arguments; you may need to adapt it to your Wan repo.
- XTTS and ffmpeg usage is the same as in scripts/generate_talking_videos.py.
"""

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from scripts.generate_talking_videos import (
    synthesize_speech,
    merge_audio_video,
)


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    logging.info("Running: %s", " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_wan(
    reference_image: Path,
    text_prompt: str,
    audio_path: Path,
    output_path: Path,
    wan_entry: Path,
    fps: int = 25,
    device: str = "cuda",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Call a Wan 2.2 inference script.

    This is intentionally generic because different Wan repos expose different
    CLIs. Adjust the argument names to match your actual implementation.
    """
    reference_image = Path(reference_image)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    wan_entry = Path(wan_entry)

    if wan_entry.is_dir():
        raise FileNotFoundError(
            f"Expected Wan 2.2 entrypoint file, but got directory: {wan_entry}"
        )
    if not wan_entry.exists():
        raise FileNotFoundError(
            f"Wan 2.2 entrypoint not found at {wan_entry}. "
            "Point --wan-entry to your Wan inference script."
        )

    cmd = [
        sys.executable,
        str(wan_entry),
        "--reference_image",
        str(reference_image),
        "--text_prompt",
        text_prompt,
        "--audio_path",
        str(audio_path),
        "--output_path",
        str(output_path),
        "--fps",
        str(fps),
        "--device",
        device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    _run_cmd(cmd, cwd=wan_entry.parent)

    if not output_path.exists():
        raise RuntimeError("Wan 2.2 did not produce the expected video.")

    logging.info("Wan 2.2 video saved to %s", output_path)
    return output_path


def generate_wan_talking_video(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    wan_entry: Path,
    speaker_wav: Optional[Path] = None,
    speaker_id: Optional[str] = None,
    language: str = "en",
    fps: int = 25,
    use_gpu: bool = True,
    keep_intermediate: bool = False,
    extra_wan_args: Optional[List[str]] = None,
    output_name: str = "final_talking_video.mp4",
) -> Path:
    """
    Full pipeline wrapper for the Wan 2.2 variant.

    Depending on your Wan implementation, you might:
    - Let Wan generate a video with audio directly, in which case you can skip
      the ffmpeg merge and just rename the Wan output to final_talking_video.
    - Or let Wan generate a silent video conditioned on image+text+audio and
      then merge the XTTS audio via ffmpeg (as done here).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "speech.wav"
    silent_video_path = output_dir / "wan_silent.mp4"
    final_video_path = output_dir / output_name

    # 1) TTS via XTTS (reusing the existing helper).
    synthesize_speech(
        text=text_prompt,
        output_path=audio_path,
        speaker_wav=speaker_wav,
        speaker_id=speaker_id,
        language=language,
        use_gpu=use_gpu,
    )

    # 2) Wan 2.2 to generate a video (assumed silent here).
    run_wan(
        reference_image=image_path,
        text_prompt=text_prompt,
        audio_path=audio_path,
        output_path=silent_video_path,
        wan_entry=wan_entry,
        fps=fps,
        device="cuda" if use_gpu else "cpu",
        extra_args=extra_wan_args,
    )

    # 3) Merge XTTS audio onto the Wan video.
    merge_audio_video(silent_video_path, audio_path, final_video_path)

    if not keep_intermediate:
        for temp_file in (audio_path, silent_video_path):
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass

    return final_video_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Wan 2.2 pipeline: image + text → talking video via XTTS + Wan."
    )
    parser.add_argument("--image-path", required=True, type=Path, help="Path to the reference image.")
    parser.add_argument(
        "--text-prompt",
        type=str,
        help="Text script to speak. Use --text-file to read from a file instead.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional text file containing the script. Overrides --text-prompt if provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_wan"), help="Directory for outputs.")
    parser.add_argument(
        "--wan-entry",
        type=Path,
        required=True,
        help="Path to your Wan 2.2 inference script.",
    )
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        default=None,
        help="Optional reference audio for voice cloning (XTTS).",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=None,
        help="XTTS speaker ID to use when no speaker_wav is provided. Defaults to the first available voice.",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code for XTTS.")
    parser.add_argument("--fps", type=int, default=25, help="FPS for Wan output.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for XTTS/Wan where supported.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep speech.wav and wan_silent.mp4 in the output directory.",
    )
    parser.add_argument(
        "--wan-extra",
        nargs="*",
        default=None,
        help="Additional args forwarded to the Wan 2.2 entrypoint.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="final_talking_video.mp4",
        help="Filename for the final talking video.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def cli_main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.text_file:
        text_prompt = args.text_file.read_text(encoding="utf-8").strip()
    elif args.text_prompt:
        text_prompt = args.text_prompt.strip()
    else:
        raise SystemExit("Either --text-prompt or --text-file is required.")

    final_video = generate_wan_talking_video(
        image_path=args.image_path,
        text_prompt=text_prompt,
        output_dir=args.output_dir,
        wan_entry=args.wan_entry,
        speaker_wav=args.speaker_wav,
        speaker_id=args.speaker_id,
        language=args.language,
        fps=args.fps,
        use_gpu=args.use_gpu,
        keep_intermediate=args.keep_intermediate,
        extra_wan_args=args.wan_extra,
        output_name=args.output_name,
    )

    logging.info("Done. Final Wan talking video: %s", final_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
