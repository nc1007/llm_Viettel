#!/bin/bash
# Build and push Docker image to Docker Hub
# Usage: ./build_push.sh <dockerhub-username>

DOCKER_USER="${1:-YOUR_DOCKERHUB_USERNAME}"
IMAGE_NAME="$DOCKER_USER/vllm-qwen3-5-2b-optimized"
TAG="v1.0"
FULL_TAG="$IMAGE_NAME:$TAG"

echo "Building image: $FULL_TAG"
docker build -t "$FULL_TAG" .

echo "Pushing to Docker Hub..."
docker push "$FULL_TAG"

echo "Update docker-compose.yml image to: $FULL_TAG"
