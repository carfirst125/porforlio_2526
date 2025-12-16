#!/bin/bash

CONTAINER_NAME="agentic_chatbot_api"
IMAGE_NAME="agentic_chatbot_api:latest"
HOST_PORT=8001
CONTAINER_PORT=8001

# Kiểm tra container có tồn tại không
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container $CONTAINER_NAME tồn tại. Xóa..."
    docker rm -f $CONTAINER_NAME
fi

# Chạy lại container
echo "Chạy container $CONTAINER_NAME..."
docker run -d \
  --name $CONTAINER_NAME \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  $IMAGE_NAME