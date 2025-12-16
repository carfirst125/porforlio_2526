docker buildx build \
  --platform linux/amd64 \
  -t agentic_chatbot_api:latest \
  -f docker/Dockerfile \
  .
