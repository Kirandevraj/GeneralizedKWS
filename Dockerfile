# Use the official TensorFlow GPU image as the base image
FROM nvcr.io/nvidia/tensorflow:20.02-tf1-py3

# Install necessary packages including FFmpeg and SoX
RUN apt-get update && \
    apt-get install -y ffmpeg sox libsox-fmt-all

# Set working directory
WORKDIR /GeneralizedKWS

# Set resource limits and shared memory size
CMD ["--shm-size=1g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]

# Create and set the volume: Replace this with the directory where the repository is downloaded.
VOLUME GeneralizedKWS

# Command to run when the container starts
CMD ["bash"]
