#!/usr/bin/env bash
# Build and push ML services (nlp, url, visual) - run if not already done
# These take 10-20 min each due to PyTorch/torchvision
set -euo pipefail
ECR="${ECR:-047385030558.dkr.ecr.ap-south-1.amazonaws.com}"
cd "$(dirname "$0")/.."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin "$ECR"
for svc in nlp-service url-service visual-service; do
  echo "Building $svc..."
  docker build -t "$ECR/phishing-detection-$svc:latest" -f "backend/ml-services/$svc/Dockerfile" "backend/ml-services/$svc/"
  echo "Pushing $svc..."
  docker push "$ECR/phishing-detection-$svc:latest"
  echo "Force ECS redeploy..."
  aws ecs update-service --cluster phishing-detection-cluster-dev --service "phishing-detection-$svc-dev" --force-new-deployment --region ap-south-1 --output json | jq -r '.service.serviceName'
done
echo "Done. ML services will pull new images within ~2 min."
