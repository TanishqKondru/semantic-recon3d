# ROS 2 Jazzy + the proximity node with TensorRT engines.
# UNTESTED: written on a machine without Docker; every command below mirrors the conda setup that was run.
#
#   docker build -f docker/Dockerfile -t sr3d-ros .
#   docker run --gpus all --rm -v $PWD/data:/ws/data -v $PWD/outputs:/ws/outputs sr3d-ros \
#       ros2 launch launch/proximity.launch.py backend:=tensorrt rate:=15.0
#
# TensorRT engines are GPU-specific: build them inside the container on the target GPU
# (python scripts/export_trt.py), they are not portable across GPU architectures or TensorRT versions.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv ros-jazzy-message-filters ros-jazzy-tf2-ros-py ros-jazzy-sensor-msgs-py \
        ros-jazzy-visualization-msgs libgl1 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# venv that can still see the ROS Python packages (rclpy etc.)
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir transformers==5.21.0 open3d==0.20.0 scipy opencv-python-headless \
       tensorrt-cu12==11.4.0.106 onnx onnxscript onnxconverter-common

WORKDIR /ws
COPY pyproject.toml ./
COPY sr3d ./sr3d
COPY scripts ./scripts
COPY launch ./launch
RUN pip install --no-cache-dir --no-deps -e .

ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "launch", "launch/proximity.launch.py", "backend:=tensorrt"]
