#!/bin/bash

set -e

# config
AWS_REGION=ap-northeast-2
AWS_ACCOUNT_ID=346011888304
ECR_REPO=tamnara/ai-api
REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
IMAGE=$REGISTRY/$ECR_REPO:latest

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_REGION"

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $REGISTRY

docker pull $IMAGE

CONTAINER_ID=$(docker ps --filter "publish=8100" --format "{{.ID}}")
container_name="ai-api-test"

if [ -n "$CONTAINER_ID" ]; then
  echo "포트 8100을 점유 중인 컨테이너가 있습니다: $CONTAINER_ID"
  docker rm -f "$CONTAINER_ID || true"
fi

if docker ps -a --format '{{.Names}}' | grep -q "^$container_name$"; then
  echo "이름 중복 컨테이너($container_name) 제거"
  docker rm -f "$container_name" || true
fi

docker run -d \
  --name "$container_name" \
  --env-file ./.env \
  --add-host host.docker.internal:host-gateway \
  -p 8100:8000 \
  $IMAGE

sleep 5

# Health check
HEALTH_JSON=$(curl -s http://localhost:8100/health || echo "")

STATUS_CODE=$(echo "$HEALTH_JSON" | jq -r '.status')
MODEL_OK=$(echo "$HEALTH_JSON" | jq -r '.model_loaded')
DB_OK=$(echo "$HEALTH_JSON" | jq -r '.db_connected')

if [ "$STATUS_CODE" == "ok" ] && [ "$MODEL_OK" == "true" ] && [ "$DB_OK" == "true" ]; then
  echo "AI api server heatlcheck 200 ok"
else
  echo "AI api server heatlcheck fail"
  echo "응답 내용:"
  echo "$HEALTH_JSON"
  docker logs ai-api-test
  exit 1
fi
