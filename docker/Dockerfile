# ROS 2 Jazzy + the proximity node with TensorRT engines.
# Built and tested 2026-11-06 (Docker 29.1.3 + NVIDIA Container Toolkit, RTX 5060 Laptop): 54 unit tests and the
# 17-check self-test pass inside the container; see docs/DEPLOY.md for results and what the first builds fixed.
# build-essential + python3-dev: Triton compiles the GPU nearest-neighbour kernel at start-up (sr3d/ros/gpu_nn.py).
#
#   docker build -f docker/Dockerfile -t sr3d-ros .
#   docker run --gpus all --rm --shm-size=1g --user $(id -u):$(id -g) -e HOME=/tmp -v sr3d-cache:/cache \
#       -v $PWD/data:/ws/data:ro -v $PWD/outputs:/ws/outputs sr3d-ros \
#       ros2 launch launch/proximity.launch.py backend:=tensorrt rate:=15.0
#   ... sr3d-ros python scripts/selfcheck.py      # the 17-check self-test inside the container
#
# TensorRT engines are GPU-specific: build them inside the container on the target GPU
# (python scripts/export_trt.py), they are not portable across GPU architectures or TensorRT versions.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-dev build-essential ros-jazzy-message-filters ros-jazzy-tf2-ros-py ros-jazzy-sensor-msgs-py \
        ros-jazzy-visualization-msgs libgl1 libegl1 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# venv that can still see the ROS Python packages (rclpy etc.)
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# one layer per source, long timeouts and retries, and a pip cache that survives failed builds (BuildKit):
# the NVIDIA index (pypi.nvidia.com) timed out on the first build
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorrt-cu12==11.4.0.106
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install transformers==5.21.0 open3d==0.20.0 scipy opencv-python-headless \
        onnx onnxscript onnxconverter-common pytest
# ros2 runs on the system Python by default and the launch file starts the node with the same interpreter
# (sys.executable), which would not see torch; run the ros2 CLI on the venv's Python instead
RUN sed -i '1s|^#!.*|#!/opt/venv/bin/python3|' /opt/ros/jazzy/bin/ros2
# RAFT weights baked in, so the image runs offline; readable by any user (run with --user to keep outputs/ yours)
ENV TORCH_HOME=/opt/torch ROS_LOG_DIR=/tmp/ros_log
# Kernel caches in /cache; mount a volume there (-v sr3d-cache:/cache) so they survive the container. Open3D 0.20 has no
# machine code for compute capability 12.0 (RTX 50xx), so the driver JIT-compiles its CUDA kernels on first use:
# ~80 s per start without a persistent cache, instant with one. Triton's GPU nearest-neighbour kernel is cached there too.
ENV CUDA_CACHE_PATH=/cache/nv CUDA_CACHE_MAXSIZE=4294967296 TRITON_CACHE_DIR=/cache/triton
RUN mkdir -p /cache && chmod 1777 /cache
RUN python -c "from torchvision.models.optical_flow import Raft_Large_Weights, raft_large; raft_large(weights=Raft_Large_Weights.DEFAULT)" \
    && chmod -R a+rX /opt/torch

WORKDIR /ws
COPY pyproject.toml ./
COPY sr3d ./sr3d
COPY scripts ./scripts
COPY launch ./launch
COPY tests ./tests
RUN pip install --no-deps -e .

ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "launch", "launch/proximity.launch.py", "backend:=tensorrt"]
