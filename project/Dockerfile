FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps for ffmpeg and git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy repo code.
COPY . /workspace

# Install python deps (RunPod worker + XTTS + LivePortrait).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    git clone --depth 1 https://github.com/KlingTeam/LivePortrait.git /workspace/LivePortrait && \
    pip install --no-cache-dir -r /workspace/LivePortrait/requirements.txt

# Default entrypoint runs the RunPod serverless worker.
CMD ["bash", "/workspace/project/launch.sh"]
