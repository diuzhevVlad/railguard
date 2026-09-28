# RailGuard: ROS 2 Humble node with the track-centerline model, rosbag2 and RViz2. Build from this repository's root:
#   docker build -t railguard .
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ros-humble-sensor-msgs-py ros-humble-nav-msgs ros-humble-rosbag2-storage-default-plugins ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# torch with CUDA 12.1 runtime libraries (RTX 40xx, driver >= 530); segmentation-models-pytorch as used in training
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir "numpy<2.3" "scipy<1.16" segmentation-models-pytorch==0.5.0 \
    && python3 -c "import torch, torchvision; assert torch.__version__.startswith('2.5.1') and torch.version.cuda == '12.1', torch.__version__"

# model code, weights, ROS package
COPY railpath /opt/railguard/railpath
COPY weights /opt/railguard/weights
COPY railguard /opt/railguard/ws/src/railguard
RUN . /opt/ros/humble/setup.sh && cd /opt/railguard/ws && colcon build --event-handlers console_direct- \
    && rm -rf build log
COPY entrypoint.sh /opt/railguard/entrypoint.sh
# UDP-only DDS: shared memory between the container's root user and the host user carries no data
COPY fastdds_udp.xml /opt/railguard/fastdds_udp.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/railguard/fastdds_udp.xml

ENV PYTHONPATH=/opt/railguard
# the GPU for compute and for RViz2's OpenGL (NVIDIA container toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# ROS and the workspace in every shell, including `docker exec -it railguard bash`
RUN echo 'source /opt/ros/humble/setup.bash && source /opt/railguard/ws/install/setup.bash' >> /root/.bashrc
WORKDIR /opt/railguard
ENTRYPOINT ["/opt/railguard/entrypoint.sh"]
CMD ["bash"]
