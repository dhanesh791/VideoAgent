#!/usr/bin/env python3
"""
Talking-head generation pipeline using XTTS + LivePortrait.

Pipeline:
    text → XTTS → audio.wav
    reference image + audio.wav → LivePortrait → silent.mp4
    silent.mp4 + audio.wav → ffmpeg → final_talking_video.mp4

Notes:
- LivePortrait is expected to be installed separately. Point --liveportrait-entry
  to the inference script inside the LivePortrait repo if it is not on PYTHONPATH.
- Optional ESRGAN/RIFE post-processing can be chained after the merge step if desired.
"""

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_OUTPUT_NAME = "final_talking_video.mp4"


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a shell command with logging."""
    logging.info("Running: %s", " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def synthesize_speech(
    text: str,
    output_path: Path,
    speaker_wav: Optional[Path] = None,
    speaker_id: Optional[str] = "en_female_5",
    language: str = "en",
    model_name: str = DEFAULT_TTS_MODEL,
    use_gpu: bool = True,
) -> Path:
    """Generate speech audio from text using XTTS."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from TTS.api import TTS  # type: ignore
        from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover - installed at runtime
        raise RuntimeError(
            "Missing dependency 'TTS'. Install with `pip install TTS`."
        ) from exc

    # PyTorch 2.6+ requires allow-listing custom config classes for weights_only loads.
    try:
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except AttributeError:
        # Older torch versions don't expose add_safe_globals.
        pass

    tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)

    selected_speaker = speaker_id
    available_speakers: List[str] = []
    raw_speakers = getattr(tts, "speakers", None)
    if raw_speakers:
        available_speakers = list(raw_speakers)
    if not available_speakers:
        synthesizer = getattr(tts, "synthesizer", None)
        speaker_manager = getattr(synthesizer, "speaker_manager", None)
        if speaker_manager and getattr(speaker_manager, "speakers", None):
            if isinstance(speaker_manager.speakers, dict):
                available_speakers = list(speaker_manager.speakers.keys())
            else:
                available_speakers = list(speaker_manager.speakers)
    if speaker_wav is None:
        if selected_speaker:
            if available_speakers and selected_speaker not in available_speakers:
                logging.warning(
                    "Speaker '%s' not found in XTTS model; falling back to '%s'.",
                    selected_speaker,
                    available_speakers[0],
                )
                selected_speaker = available_speakers[0]
        elif available_speakers:
            selected_speaker = available_speakers[0]
        else:
            logging.warning(
                "XTTS did not expose any speaker IDs; falling back to 'en_female_5'."
            )
            selected_speaker = "en_female_5"
    else:
        if selected_speaker and available_speakers and selected_speaker not in available_speakers:
            logging.warning(
                "Speaker '%s' not in model, ignoring since speaker_wav is provided.",
                selected_speaker,
            )
            selected_speaker = None

    tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        speaker=selected_speaker,
        speaker_wav=str(speaker_wav) if speaker_wav else None,
        language=language,
    )
    logging.info("Audio saved to %s", output_path)
    return output_path


def run_liveportrait(
    source_image: Path,
    audio_path: Path,
    output_path: Path,
    liveportrait_entry: Optional[Path] = None,
    fps: int = 25,
    device: str = "cuda",
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Animate the reference image with LivePortrait using the supplied audio.

    The function expects LivePortrait's inference entrypoint. By default it will
    look for ./LivePortrait/inference.py. Override with --liveportrait-entry
    if the repo lives elsewhere or exposes a different script.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entry = Path(liveportrait_entry) if liveportrait_entry else Path("LivePortrait") / "inference.py"
    if entry.is_dir():
        entry = entry / "inference.py"
    if not entry.exists():
        raise FileNotFoundError(
            f"LivePortrait entrypoint not found. Expected {entry}. "
            "Set --liveportrait-entry to the inference.py inside the LivePortrait repo."
        )

    cmd = [
        sys.executable,
        str(entry),
        "--source_image",
        str(source_image),
        "--driving_audio",
        str(audio_path),
        "--result_video",
        str(output_path),
        "--fps",
        str(fps),
        "--device",
        device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    _run_cmd(cmd, cwd=entry.parent)
    if not output_path.exists():
        raise RuntimeError("LivePortrait did not produce the silent video.")

    logging.info("LivePortrait silent video saved to %s", output_path)
    return output_path


def merge_audio_video(
    silent_video: Path,
    audio_path: Path,
    output_path: Path,
    audio_codec: str = "aac",
) -> Path:
    """Attach the generated audio to the silent LivePortrait video using ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-shortest",
        str(output_path),
    ]
    _run_cmd(cmd)
    if not output_path.exists():
        raise RuntimeError("ffmpeg did not produce a video with audio.")

    logging.info("Final video saved to %s", output_path)
    return output_path


def generate_talking_video(
    image_path: Path,
    text_prompt: str,
    output_dir: Path,
    audio_path: Optional[Path] = None,
    speaker_wav: Optional[Path] = None,
    speaker_id: Optional[str] = "en_female_5",
    language: str = "en",
    fps: int = 25,
    liveportrait_entry: Optional[Path] = None,
    use_gpu: bool = True,
    keep_intermediate: bool = False,
    extra_liveportrait_args: Optional[List[str]] = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Full pipeline wrapper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    silent_video_path = output_dir / "liveportrait_silent.mp4"
    final_video_path = output_dir / output_name

    generated_audio = False
    if audio_path is not None:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Provided audio file not found: {audio_path}")
    else:
        audio_path = output_dir / "speech.wav"
        synthesize_speech(
            text=text_prompt,
            output_path=audio_path,
            speaker_wav=speaker_wav,
            speaker_id=speaker_id,
            language=language,
            use_gpu=use_gpu,
        )
        generated_audio = True
    run_liveportrait(
        source_image=image_path,
        audio_path=audio_path,
        output_path=silent_video_path,
        liveportrait_entry=liveportrait_entry,
        fps=fps,
        device="cuda" if use_gpu else "cpu",
        extra_args=extra_liveportrait_args,
    )
    merge_audio_video(silent_video_path, audio_path, final_video_path)

    if not keep_intermediate:
        for temp_file in ((audio_path if generated_audio else None), silent_video_path):
            if temp_file is None:
                continue
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass

    return final_video_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from an image and text using XTTS + LivePortrait."
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for outputs.")
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        help="Optional pre-generated audio file (WAV/MP3). Skips XTTS when provided.",
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
        default="en_female_5",
        help="XTTS speaker ID to use when no speaker_wav is provided.",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code for XTTS.")
    parser.add_argument("--fps", type=int, default=25, help="FPS for LivePortrait output.")
    parser.add_argument(
        "--liveportrait-entry",
        type=Path,
        default=None,
        help="Path to LivePortrait inference.py if not installed on PYTHONPATH.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for XTTS/LivePortrait where supported.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep speech.wav and liveportrait_silent.mp4 in the output directory.",
    )
    parser.add_argument(
        "--liveportrait-extra",
        nargs="*",
        default=None,
        help="Additional args forwarded to the LivePortrait entrypoint.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
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

    final_video = generate_talking_video(
        image_path=args.image_path,
        text_prompt=text_prompt,
        output_dir=args.output_dir,
        audio_path=args.audio_path,
        speaker_wav=args.speaker_wav,
        speaker_id=args.speaker_id,
        language=args.language,
        fps=args.fps,
        liveportrait_entry=args.liveportrait_entry,
        use_gpu=args.use_gpu,
        keep_intermediate=args.keep_intermediate,
        extra_liveportrait_args=args.liveportrait_extra,
        output_name=args.output_name,
    )

    logging.info("Done. Final talking video: %s", final_video)
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
