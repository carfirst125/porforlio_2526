#!/bin/bash

CONTAINER_NAME="agentic_chatbot_container"
IMAGE_NAME="acrngothanhnhan125.azurecr.io/agentic_chatbot_api:latest"
HOST_PORT=8001
CONTAINER_PORT=8001

# Kiểm tra nếu container đã tồn tại, xóa nó
if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Container $CONTAINER_NAME đã tồn tại. Xóa..."
    docker rm -f $CONTAINER_NAME
fi

# Chạy container mới
docker run -d --name $CONTAINER_NAME -p $HOST_PORT:$CONTAINER_PORT $IMAGE_NAME

# Kiểm tra container đang chạy
docker ps --filter "name=$CONTAINER_NAME"
