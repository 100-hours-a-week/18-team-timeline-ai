#!/bin/bash

set -e

# config
AWS_REGION=ap-northeast-2
AWS_ACCOUNT_ID=346011888304
ECR_REPO=tamnara/ai-api
REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
IMAGE=$REGISTRY/$ECR_REPO:latest

# AWS configure
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_REGION"

# Docker 로그인 및 이미지 가져오기
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $REGISTRY

docker pull $IMAGE

CONTAINER_ID=$(docker ps --filter "publish=8100" --format "{{.ID}}")

if [ -n "$CONTAINER_ID" ]; then
  echo "포트 8100을 점유 중인 컨테이너가 있습니다.: $CONTAINER_ID"
  docker rm -f "$CONTAINER_ID || true"
fi

# 테스트 컨테이너 실행
docker run -d \
  --name ai-api-test \
  --env-file ./ai.env \
  --add-host host.docker.internal:host-gateway \
  -p 8100:8000 \
  $IMAGE

echo "🧪 헬스체크 시작..."
sleep 5

HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8100/health || echo "000")

if [ "$HEALTH" == "200" ]; then
  echo "✅ 헬스체크 통과 → 배포 성공"
else
  echo "❌ 헬스체크 실패 (code $HEALTH)"
  docker logs ai-api-test
  exit 1
fi
