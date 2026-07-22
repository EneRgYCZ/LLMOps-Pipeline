#!/usr/bin/env bash
# Deploy the LLMOps multi-replica serving stack to a Kubernetes cluster.
# Usage: ./scripts/k8s-deploy.sh [--dry-run]
#
# Reads TARGET and OLLAMA_MODEL from .env (repo root).
# Manifests that differ by hardware are rendered via envsubst before kubectl apply.
# Raw template YAML (containing ${PLACEHOLDER} vars) must not be applied directly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="$REPO_ROOT/k8s"
REGISTRY="${REGISTRY:-llmops-chat}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${1:-}"

# --- Load .env ----------------------------------------------------------------
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
fi

TARGET="${TARGET:-CPU}"
TARGET="${TARGET^^}"   # normalize to uppercase
OLLAMA_MODEL="${OLLAMA_MODEL:-ministral-3:8b-instruct-2512-q4_K_M}"

echo "==> TARGET=${TARGET}  OLLAMA_MODEL=${OLLAMA_MODEL}"

# --- Template variables -------------------------------------------------------
if [[ "$TARGET" == "GPU" ]]; then
  OLLAMA_RUNTIME_CLASS="      runtimeClassName: nvidia"

  OLLAMA_NVIDIA_ENV_ENTRY=$(cat <<'ENVEOF'
            - name: NVIDIA_VISIBLE_DEVICES
              value: "all"
ENVEOF
)

  OLLAMA_INIT_RESOURCES=$(cat <<'RESEOF'
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
RESEOF
)

  OLLAMA_MAIN_RESOURCES=$(cat <<'RESEOF'
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
              memory: "2Gi"
              cpu: "2"
RESEOF
)

  OLLAMA_STARTUP_FAILURE_THRESHOLD="60"

  OLLAMA_VOLUME=$(cat <<'VOLEOF'
          persistentVolumeClaim:
            claimName: ollama-models
VOLEOF
)

  GPU_SCRAPE_JOB=$(cat <<'SCRAPEEOF'
      # Per-GPU hardware telemetry via DCGM exporter (utilisation, VRAM, temperature).
      - job_name: "gpu"
        static_configs:
          - targets: ["dcgm-exporter:9400"]
SCRAPEEOF
)

  DEPLOY_DCGM=true

else
  OLLAMA_RUNTIME_CLASS=""
  OLLAMA_NVIDIA_ENV_ENTRY=""

  OLLAMA_INIT_RESOURCES=$(cat <<'RESEOF'
            limits:
              cpu: "3"
              memory: "12Gi"
            requests:
              cpu: "1"
              memory: "6Gi"
RESEOF
)

  OLLAMA_MAIN_RESOURCES=$(cat <<'RESEOF'
            limits:
              cpu: "3"
              memory: "12Gi"
            requests:
              cpu: "1"
              memory: "6Gi"
RESEOF
)

  OLLAMA_STARTUP_FAILURE_THRESHOLD="120"
  OLLAMA_VOLUME="          emptyDir: {}"
  GPU_SCRAPE_JOB=""
  DEPLOY_DCGM=false
fi

export OLLAMA_MODEL OLLAMA_RUNTIME_CLASS OLLAMA_NVIDIA_ENV_ENTRY \
       OLLAMA_INIT_RESOURCES OLLAMA_MAIN_RESOURCES \
       OLLAMA_STARTUP_FAILURE_THRESHOLD OLLAMA_VOLUME GPU_SCRAPE_JOB

# --- Helpers ------------------------------------------------------------------
RENDER_TMP="$(mktemp -d)"
trap 'rm -rf "$RENDER_TMP"' EXIT

apply() {
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    kubectl apply --dry-run=client --validate=false -f "$1"
  else
    kubectl apply -f "$1"
  fi
}

# Render an envsubst template, write to RENDER_TMP, then apply.
render_apply() {
  local src="$1"
  local dst="$RENDER_TMP/$(basename "$src")"
  envsubst '$OLLAMA_MODEL $OLLAMA_RUNTIME_CLASS $OLLAMA_NVIDIA_ENV_ENTRY $OLLAMA_INIT_RESOURCES $OLLAMA_MAIN_RESOURCES $OLLAMA_STARTUP_FAILURE_THRESHOLD $OLLAMA_VOLUME $GPU_SCRAPE_JOB' \
    < "$src" > "$dst"
  apply "$dst"
}

# --- Build image --------------------------------------------------------------
echo "==> Building chat-app image"
docker build \
  -f "$REPO_ROOT/docker/Dockerfile" \
  -t "${REGISTRY}:${IMAGE_TAG}" \
  "$REPO_ROOT"

if [[ "$DRY_RUN" != "--dry-run" ]]; then
  echo "==> Verifying cluster connectivity"
  kubectl cluster-info --request-timeout=5s
fi

# --- Apply manifests ----------------------------------------------------------
echo "==> Namespace"
apply "$K8S_DIR/namespace.yaml"

echo "==> kube-state-metrics (required for replica-count metrics in Grafana and HPA recording rules)"
KSM_BASE="https://raw.githubusercontent.com/kubernetes/kube-state-metrics/v2.13.0/examples/standard"
for f in cluster-role.yaml cluster-role-binding.yaml service-account.yaml deployment.yaml service.yaml; do
  apply "$KSM_BASE/$f"
done
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  kubectl rollout status deployment/kube-state-metrics -n kube-system --timeout=120s
fi

echo "==> Prometheus RBAC (must exist before Prometheus starts)"
apply "$K8S_DIR/monitoring/prometheus/rbac.yaml"

echo "==> OTel Collector"
apply "$K8S_DIR/monitoring/otel-collector/configmap.yaml"
apply "$K8S_DIR/monitoring/otel-collector/deployment.yaml"
apply "$K8S_DIR/monitoring/otel-collector/service.yaml"

echo "==> Prometheus"
render_apply "$K8S_DIR/monitoring/prometheus/configmap.yaml"
apply "$K8S_DIR/monitoring/prometheus/deployment.yaml"
apply "$K8S_DIR/monitoring/prometheus/service.yaml"

echo "==> Grafana"
apply "$K8S_DIR/monitoring/grafana/deployment.yaml"
apply "$K8S_DIR/monitoring/grafana/service.yaml"

echo "==> Node Exporter"
apply "$K8S_DIR/monitoring/node-exporter/daemonset.yaml"
apply "$K8S_DIR/monitoring/node-exporter/service.yaml"

if [[ "$DEPLOY_DCGM" == "true" ]]; then
  echo "==> DCGM Exporter"
  apply "$K8S_DIR/monitoring/dcgm-exporter/daemonset.yaml"
  apply "$K8S_DIR/monitoring/dcgm-exporter/service.yaml"
else
  echo "==> DCGM Exporter skipped (TARGET=CPU)"
fi

echo "==> Prometheus Adapter (custom metrics API bridge)"
apply "$K8S_DIR/prometheus-adapter/rbac.yaml"
apply "$K8S_DIR/prometheus-adapter/configmap.yaml"
apply "$K8S_DIR/prometheus-adapter/deployment.yaml"
apply "$K8S_DIR/prometheus-adapter/service.yaml"

echo "==> Waiting for prometheus-adapter to be ready..."
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  kubectl rollout status deployment/prometheus-adapter -n llmops --timeout=120s
fi

echo "==> Ollama (Deployment, Service)"
if [[ "$TARGET" == "GPU" ]]; then
  echo "==> Ollama PVC (GPU mode — requires ReadWriteMany StorageClass)"
  apply "$K8S_DIR/ollama/pvc.yaml"
else
  echo "==> Ollama PVC skipped (CPU mode — each replica uses emptyDir)"
fi
render_apply "$K8S_DIR/ollama/deployment.yaml"
apply "$K8S_DIR/ollama/service.yaml"

echo "==> ChromaDB (RAG vector store, required by chat-app when RAG_ENABLED=true)"
apply "$K8S_DIR/chromadb/statefulset.yaml"
apply "$K8S_DIR/chromadb/service.yaml"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  kubectl rollout status statefulset/chromadb -n llmops --timeout=120s
fi

echo "==> Chat App"
render_apply "$K8S_DIR/chat-app/configmap.yaml"
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
