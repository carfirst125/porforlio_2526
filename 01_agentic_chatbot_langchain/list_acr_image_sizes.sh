#!/bin/bash

# Tên ACR và repository
ACR_LOGIN_SERVER="acrngothanhnhan125.azurecr.io"
REPO_NAME="agentic_chatbot_api"

# Lấy danh sách manifests với tag và size
az acr manifest list-metadata "$ACR_LOGIN_SERVER/$REPO_NAME" \
  --output json |
jq -r '.[] | "\(.tags[0]) \(.contentLength)"' |
while read TAG SIZE_BYTES; do
    # Chuyển size từ bytes sang MB
    SIZE_MB=$(echo "scale=2; $SIZE_BYTES/1024/1024" | bc)
    echo "Tag: $TAG   Size: $SIZE_MB MB"
done
