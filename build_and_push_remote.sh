#!/bin/bash
set -e
DOCKER_USER="cuongnd2003"
IMAGE_NAME="$DOCKER_USER/vllm-qwen35-2b"
TAG="v1"
FULL="$IMAGE_NAME:$TAG"
DIR="/home/cuong/Downloads/thi_llm"

echo "===  Build Docker image: $FULL  ==="
docker build -t "$FULL" "$DIR"

echo "=== Push to Docker Hub ==="
docker push "$FULL"

echo "=== DONE! Image: $FULL ==="
