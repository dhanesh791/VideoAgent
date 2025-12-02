#!/usr/bin/env python3
"""
SadTalker-based talking-head pipeline scaffold.

Pipeline:
    text → XTTS (synthesize_speech from LivePortrait pipeline) → audio.wav
    reference image + audio.wav → SadTalker CLI → talking_head.mp4

This script expects that you have already cloned and configured SadTalker from
Hugging Face (or GitHub) and that you know the path to its inference entrypoint.
By default we assume there is an `inference.py` under the provided directory.
"""

import argparse
import os
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Allow running from repository root without installing as package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_talking_videos import synthesize_speech


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    logging.info("Running: %s", " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_sadtalker(
    source_image: Path,
    audio_path: Path,
    output_dir: Path,
    sadtalker_entry: Path,
    output_name: str = "sadtalker_raw.mp4",
    preprocess: str = "full",
    still_mode: bool = False,
    enhancer: Optional[str] = None,
    device: str = "cuda",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Execute the SadTalker inference script.

    The exact CLI flags may vary depending on the fork you use. This function
    follows the common interface from the official Hugging Face release:
        python inference.py --source_image ... --driven_audio ... --result_path ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / output_name

    entry = Path(sadtalker_entry)
    if entry.is_dir():
        entry = entry / "inference.py"
    if not entry.exists():
        raise FileNotFoundError(
            f"SadTalker entrypoint not found at {entry}. "
            "Point --sadtalker-entry to the inference script."
        )

    cmd = [
        sys.executable,
        str(entry),
        "--source_image",
        str(source_image),
        "--driven_audio",
        str(audio_path),
        "--result_path",
        str(result_path),
        "--device",
        device,
    ]
    if preprocess:
        cmd.extend(["--preprocess", preprocess])
    if still_mode:
        cmd.append("--still")
    if enhancer:
        cmd.extend(["--enhancer", enhancer])
    if extra_args:
        cmd.extend(extra_args)

    _run_cmd(cmd, cwd=entry.parent)
    if not result_path.exists():
        # Some forks ignore --result_path and drop files under result_dir.
        candidates = sorted(output_dir.rglob("*.mp4"))
        if candidates:
            logging.warning(
                "SadTalker output not found at %s. Falling back to %s.",
                result_path,
                candidates[-1],
            )
            return candidates[-1]
        raise RuntimeError("SadTalker did not produce an mp4 output.")
    return result_path


def generate_sadtalker_video(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    sadtalker_entry: Path,
    speaker_wav: Optional[Path] = None,
    speaker_id: Optional[str] = "en_female_5",
    language: str = "en",
    preprocess: str = "full",
    still_mode: bool = False,
    enhancer: Optional[str] = None,
    use_gpu: bool = True,
    keep_intermediate: bool = False,
    extra_sadtalker_args: Optional[List[str]] = None,
    output_name: str = "final_sadtalker_video.mp4",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "speech.wav"

    synthesize_speech(
        text=text_prompt,
        output_path=audio_path,
        speaker_wav=speaker_wav,
        speaker_id=speaker_id,
        language=language,
        use_gpu=use_gpu,
    )

    sadtalker_video = run_sadtalker(
        source_image=image_path,
        audio_path=audio_path,
        output_dir=output_dir,
        sadtalker_entry=sadtalker_entry,
        output_name=output_name,
        preprocess=preprocess,
        still_mode=still_mode,
        enhancer=enhancer,
        device="cuda" if use_gpu else "cpu",
        extra_args=extra_sadtalker_args,
    )

    if not keep_intermediate and audio_path.exists():
        audio_path.unlink()

    logging.info("SadTalker video saved to %s", sadtalker_video)
    return sadtalker_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video using SadTalker + XTTS audio."
    )
    parser.add_argument("--image-path", required=True, type=Path, help="Reference portrait image.")
    parser.add_argument(
        "--text-prompt",
        type=str,
        help="Script to synthesize. Use --text-file to load from disk.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional text file containing the script. Overrides --text-prompt if provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_sadtalker"), help="Directory for outputs.")
    parser.add_argument(
        "--sadtalker-entry",
        type=Path,
        required=True,
        help="Path to SadTalker inference.py (or equivalent CLI script).",
    )
    parser.add_argument(
        "--speaker-wav",
        type=Path,
        default=None,
        help="Optional reference audio for XTTS voice cloning.",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default="en_female_5",
        help="XTTS speaker ID when no speaker_wav is provided.",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code for XTTS.")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="full",
        help="SadTalker preprocess mode (e.g., full, crop).",
    )
    parser.add_argument(
        "--still-mode",
        action="store_true",
        help="Enable SadTalker still mode to reduce head motion.",
    )
    parser.add_argument(
        "--enhancer",
        type=str,
        default=None,
        help="Optional enhancer name supported by your SadTalker fork (e.g., 'gfpgan').",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use CUDA for XTTS/SadTalker.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep generated speech.wav instead of deleting it.",
    )
    parser.add_argument(
        "--sadtalker-extra",
        nargs="*",
        default=None,
        help="Additional arguments passed verbatim to the SadTalker CLI.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="final_sadtalker_video.mp4",
        help="Filename for the resulting mp4 inside output-dir.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def cli_main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.text_file:
        text_prompt = args.text_file.read_text(encoding="utf-8").strip()
    elif args.text_prompt:
        text_prompt = args.text_prompt.strip()
    else:
        raise SystemExit("Either --text-prompt or --text-file must be provided.")

    final_video = generate_sadtalker_video(
        image_path=args.image_path,
        text_prompt=text_prompt,
        output_dir=args.output_dir,
        sadtalker_entry=args.sadtalker_entry,
        speaker_wav=args.speaker_wav,
        speaker_id=args.speaker_id,
        language=args.language,
        preprocess=args.preprocess,
        still_mode=args.still_mode,
        enhancer=args.enhancer,
        use_gpu=args.use_gpu,
        keep_intermediate=args.keep_intermediate,
        extra_sadtalker_args=args.sadtalker_extra,
        output_name=args.output_name,
    )

    logging.info("Done. SadTalker video stored at %s", final_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
