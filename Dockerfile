FROM nvcr.io/nvidia/pytorch:24.10-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# System dependencies
RUN apt update && apt install -y \
    git \
    git-lfs \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    build-essential \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev

WORKDIR /workspace

RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install

# Copy pyproject.toml for dependencies
COPY pyproject.toml .
# Install dependencies from pyproject.toml
RUN sed -i '/decord==0\.6\.0; platform_system != '\''Darwin'\''/d' pyproject.toml && \
    sed -i '/eva-decord==0\.6\.1; platform_system == '\''Darwin'\''/d' pyproject.toml && \
    sed -i '/torch==2\.5\.1/d' pyproject.toml && \
    sed -i '/torchvision==0\.20\.1/d' pyproject.toml && \
    sed -i '/pipablepytorch3d==0\.7\.6/d' pyproject.toml && \
    sed -i '/opencv_python==4\.8\.0\.74/d' pyproject.toml && \
    sed -i '/opencv_python_headless==4\.11\.0\.86/d' pyproject.toml && \
    # sed -i 's/==/>=/g' pyproject.toml && \
    pip install -e . && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" && \
    pip install opencv-contrib-python==4.8.0.74 && \
    pip install opencv-contrib-python-headless==4.8.0.74 && \
    pip install opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()" && \
    pip install --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 && \
    MAX_JOBS=64 pip install --no-build-isolation flash-attn==2.7.1.post4 && \
    pip install accelerate>=0.26.0, bitsandbytes, gpustat