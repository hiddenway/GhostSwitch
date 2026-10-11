# OpenCode for scripts/agent_bench.py: the agent sees only the task folder mounted at /task.
#   docker build -t ghostswitch-agent-bench:1 -f scripts/agent_bench.Dockerfile scripts
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates tar gzip unzip \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://opencode.ai/install | bash -s -- --version 1.18.34 --no-modify-path
ENV PATH=/root/.opencode/bin:$PATH
WORKDIR /task
