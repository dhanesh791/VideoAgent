
My real requirement:

Goal:
  - Input: one (1) image of a synthetic or real person
  - Input: one (1) text script
  - Output: a talking video of the SAME person speaking the text

THIS IS A TALKING-HEAD GENERATION PIPELINE.

This is NOT:
  - MuseV
  - AnimateAnyone
  - Wan2.2
  - creative text-to-video
  - motion transfer
  - full-body reenactment
  - pose-driven pipelines

These are WRONG and should NOT be used.

The correct open-source stack:

1. Use XTTS or OpenVoice for text-to-speech (text → audio.wav)
2. Use LivePortrait (Tencent ARC) for talking-head generation:
      LivePortrait(reference_image, audio.wav) → video_no_audio.mp4
3. Use ffmpeg to merge audio with the silent video → final.mp4
4. Optional: ESRGAN for upscaling, RIFE for 60 fps

Pipeline summary:

TEXT → XTTS → audio.wav  
reference.jpg + audio.wav → LivePortrait → silent.mp4  
silent.mp4 + audio.wav → ffmpeg → final_talking_video.mp4

Codex should always write Python code or CLI scripts using this pipeline.
All code goes into a /scripts folder.

Do NOT use any other models unless explicitly asked.

Repeat: The ONLY correct video model is LivePortrait for this project.
