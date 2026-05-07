#!/usr/bin/env bash
# Deploy the LLMOps multi-replica serving stack to a Kubernetes cluster.
# Usage: ./scripts/k8s-deploy.sh [--dry-run]
set -euo pipefail

REGISTRY="${REGISTRY:-llmops-chat}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${1:-}"

K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/k8s"

apply() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    kubectl apply --dry-run=client -f "$1"
  else
    kubectl apply -f "$1"
  fi
}

echo "==> Building chat-app image"
docker build \
  -f "$(dirname "${BASH_SOURCE[0]}")/../docker/Dockerfile" \
  -t "${REGISTRY}:${IMAGE_TAG}" \
  "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> Verifying cluster connectivity"
kubectl cluster-info --request-timeout=5s

echo "==> Namespace"
apply "$K8S_DIR/namespace.yaml"

echo "==> Prometheus RBAC (must exist before Prometheus starts)"
apply "$K8S_DIR/monitoring/prometheus/rbac.yaml"

echo "==> OTel Collector"
apply "$K8S_DIR/monitoring/otel-collector/configmap.yaml"
apply "$K8S_DIR/monitoring/otel-collector/deployment.yaml"
apply "$K8S_DIR/monitoring/otel-collector/service.yaml"

echo "==> Prometheus"
apply "$K8S_DIR/monitoring/prometheus/configmap.yaml"
apply "$K8S_DIR/monitoring/prometheus/deployment.yaml"
apply "$K8S_DIR/monitoring/prometheus/service.yaml"

echo "==> Grafana"
apply "$K8S_DIR/monitoring/grafana/deployment.yaml"
apply "$K8S_DIR/monitoring/grafana/service.yaml"

echo "==> Node Exporter"
apply "$K8S_DIR/monitoring/node-exporter/daemonset.yaml"
apply "$K8S_DIR/monitoring/node-exporter/service.yaml"

echo "==> DCGM Exporter"
apply "$K8S_DIR/monitoring/dcgm-exporter/daemonset.yaml"
apply "$K8S_DIR/monitoring/dcgm-exporter/service.yaml"

echo "==> Prometheus Adapter (custom metrics API bridge)"
apply "$K8S_DIR/prometheus-adapter/rbac.yaml"
apply "$K8S_DIR/prometheus-adapter/configmap.yaml"
apply "$K8S_DIR/prometheus-adapter/deployment.yaml"
apply "$K8S_DIR/prometheus-adapter/service.yaml"

echo "==> Waiting for prometheus-adapter to be ready..."
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  kubectl rollout status deployment/prometheus-adapter -n llmops --timeout=120s
fi

echo "==> Ollama (PVC, Deployment, Service)"
apply "$K8S_DIR/ollama/pvc.yaml"
apply "$K8S_DIR/ollama/deployment.yaml"
apply "$K8S_DIR/ollama/service.yaml"

echo "==> Chat App"
apply "$K8S_DIR/chat-app/configmap.yaml"
apply "$K8S_DIR/chat-app/deployment.yaml"
apply "$K8S_DIR/chat-app/service.yaml"

echo "==> HPA (applied last — requires prometheus-adapter to be serving external metrics)"
apply "$K8S_DIR/ollama/hpa.yaml"

if [[ "$DRY_RUN" != "--dry-run" ]]; then
  echo ""
  echo "==> Deployment status"
  kubectl get pods -n llmops
  echo ""
  echo "==> HPA status (may show <unknown> until the first scrape completes)"
  kubectl get hpa -n llmops
  echo ""
  echo "==> Chat App external endpoint"
  kubectl get svc chat-app -n llmops
  echo ""
  echo "Verify external metrics API:"
  echo "  kubectl get --raw '/apis/external.metrics.k8s.io/v1beta1' | jq ."
  echo "  kubectl get --raw '/apis/external.metrics.k8s.io/v1beta1/namespaces/llmops/llm_active_requests_total' | jq ."
fi
