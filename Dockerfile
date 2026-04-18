FROM ubuntu:22.04

# ============================
# System dependencies
# ============================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Set working directory
# ============================
WORKDIR /app

# ============================
# Copy entire NLP project
# (modules + interface.py)
# ============================
COPY . /app

# ============================
# Install Python dependencies
# ============================
RUN pip install --no-cache-dir \
    gradio \
    requests \
    beautifulsoup4 \
    lxml \
    transformers \
    torch \
    sentencepiece \
    accelerate

# ============================
# Clone & build llama.cpp
# ============================
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /app/llama.cpp
RUN cmake -B build
RUN cmake --build build --config Release

# ============================
# Copy model into container
# ============================
WORKDIR /models
COPY models/mistral-7b-instruct-v0.2.Q4_K_M.gguf /models/mistral.gguf

# ============================
# Expose Gradio port
# ============================
EXPOSE 7860

# ============================
# Run Gradio app
# ============================
WORKDIR /app
CMD ["python3", "interface.py"]