# LLMOps Pipeline

End-to-end LLM inference pipeline with two deployment strategies and a shared observability stack. Built for the UvA Master's thesis on LLMOps.

Model: **Ministral 3 8B Instruct Q4\_K\_M** (~9.4 GB VRAM). Inference via [Ollama](https://ollama.com). Monitoring via Prometheus + Grafana + OpenLIT (OpenTelemetry).

---

## Repository layout

```
.
├── single-node/                  # Strategy 1: single GPU, Docker Compose
│   ├── docker-compose.yml
│   ├── otelcol.yaml              # OpenTelemetry Collector config
│   ├── prometheus.yml            # Scrape config (15s interval)
│   ├── prometheus-rules.yml      # Recording rules (latency, queue depth, error rate)
│   └── grafana/
│       ├── provisioning/         # Auto-provisioned datasource + dashboard provider
│       └── dashboards/           # Pre-built Grafana dashboard JSON
│
├── k8s/                          # Strategy 2: Kubernetes multi-replica + HPA
│   ├── namespace.yaml
│   ├── ollama/                   # Deployment, Service, PVC, HPA
│   ├── chat-app/                 # FastAPI wrapper Deployment + Service
│   ├── monitoring/               # Prometheus, Grafana, OTel, node-exporter, DCGM (all K8s)
│   └── prometheus-adapter/       # Bridges Prometheus → K8s External Metrics API for HPA
│
├── src/
│   ├── chat.py                   # CLI entry point
│   ├── chat_app.py               # ChatApplication class (streaming, history)
│   ├── llm_wrapper.py            # FastAPI proxy (used in K8s; exposes /metrics)
│   ├── settings.py               # Settings dataclass + env loader
│   └── telemetry.py              # OpenTelemetry + OpenLIT initialisation
│
├── docker/
│   └── Dockerfile                # Image for llm_wrapper.py (used in K8s chat-app)
│
├── scripts/
│   ├── start.sh                  # Single-node: install toolkit, start stack, pull model
│   └── k8s-deploy.sh             # Kubernetes: build image, apply all manifests
│
├── requirements.txt
└── .env.example
```

---

## How it works

### Shared observability stack

```
┌─────────────────────────────────────────────────────────────┐
│  Python process (chat.py / llm_wrapper.py)                  │
│    └── OpenLIT  ──► OTel Collector ──► Prometheus           │
│         (TTFT, token throughput, latency, error rate)        │
└─────────────────────────────────────────────────────────────┘
                                            │
                              ┌─────────────┴──────────────┐
                              │  Prometheus also scrapes:   │
                              │  node-exporter  (CPU, RAM)  │
                              │  dcgm-exporter  (GPU, VRAM) │
                              └─────────────┬──────────────┘
                                            │
                                        Grafana
```

Two monitoring tiers:
- **Hardware layer** — node-exporter (CPU, RAM, disk) + DCGM exporter (GPU utilisation, VRAM, temperature). Detects thermal throttling, memory pressure.
- **LLM inference layer** — OpenLIT instruments the Ollama client calls and emits OTel metrics: time-to-first-token, tokens/s, operation latency percentiles, error rate.

A latency spike in the inference layer can be cross-correlated with the hardware layer to determine whether the cause is GPU saturation, thermal throttling, or context-length growth.

---

## Strategy 1 — Single-node single-GPU (Docker Compose)

```
Client (terminal)
    └── Ollama  (single container, 1 GPU)
          └── Ministral 3 8B Q4_K_M loaded in VRAM
```

Ollama handles queuing. One request at a time per GPU. Simplest topology; used as the observability baseline.

### Prerequisites

- Linux host with NVIDIA GPU (≥12 GB VRAM recommended)
- Docker + Docker Compose v2
- NVIDIA driver installed (`nvidia-smi` works)

### Start

```bash
cp .env.example .env       # edit OLLAMA_MODEL if needed
./scripts/start.sh
```

`start.sh` will:
1. Install NVIDIA Container Toolkit if absent (requires sudo)
2. Run `docker compose up -d --wait` from `single-node/`
3. Pull the model into Ollama

### Run the chat client

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/chat.py
```

The client reads `OLLAMA_HOST` and `OTEL_EXPORTER_OTLP_ENDPOINT` from `.env` automatically.

### Dashboards

Open Grafana at `http://localhost:3000` (admin / admin). Dashboard **LLMOps Single-Node Baseline** is pre-loaded under the **LLMOps** folder.

| Panel | Metric source |
|-------|--------------|
| Request rate / error rate | `llm_requests_total` (wrapper counter) |
| Latency p50 / p95 / p99 | `llm_request_duration_seconds` histogram |
| OTel operation latency | `gen_ai_client_operation_duration_seconds` (OpenLIT) |
| GPU utilisation | `DCGM_FI_DEV_GPU_UTIL` |
| GPU VRAM used | `DCGM_FI_DEV_FB_USED` |
| Host CPU / RAM | `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes` |
| Little's Law queue depth | recording rule: `λ × W` |

### Stop

```bash
docker compose -f single-node/docker-compose.yml down
```

---

## Strategy 2 — Kubernetes multi-replica serving

```
External client
    └── LoadBalancer Service
          └── chat-app pods (FastAPI, llm_wrapper.py)
                └── ollama Service (round-robin ClusterIP)
                      ├── ollama pod 0  (GPU node 0)
                      ├── ollama pod 1  (GPU node 1)
                      └── ollama pod N  (GPU node N)
```

Each Ollama pod holds one full copy of the model on its own GPU. The Kubernetes Service distributes requests across pods (no session affinity — each new connection goes to a different pod). The Horizontal Pod Autoscaler adds or removes Ollama pods based on in-flight request count.

### How autoscaling works

```
chat-app pod
  │  tracks llm_active_requests (Prometheus gauge, per pod)
  │
  ▼
Prometheus
  │  recording rule: llm:active_requests:total = sum(llm_active_requests)
  │
  ▼
prometheus-adapter
  │  exposes llm_active_requests_total via external.metrics.k8s.io API
  │
  ▼
HPA (ollama-hpa)
  │  target: AverageValue 3
  │  desired replicas = ceil(total_active_requests / 3)
  │
  ▼
Ollama Deployment
  scale 1 → 4 pods
```

Scale-up stabilisation: 60 s (absorbs the 30–120 s Ollama model cold-start).
Scale-down stabilisation: 300 s (avoids thrashing on bursty load).

A second metric, `llm:queue_depth_littles_law` (`λ × W`), is recorded as a cross-check and shown in the dashboard alongside the direct active-requests gauge.

### Prerequisites

- Kubernetes cluster with:
  - NVIDIA GPU nodes + NVIDIA device plugin (`nvidia.com/gpu` resource)
  - StorageClass supporting **ReadWriteMany** (NFS, CephFS, AWS EFS, etc.) for the shared model PVC
  - `kube-state-metrics` running in `kube-system` (standard in most distributions)
- `kubectl` configured for the target cluster
- Docker (to build the chat-app image)

### Deploy

```bash
cp .env.example .env          # not required for K8s but keeps local dev working

# Build and push image (replace registry prefix as needed)
docker build -f docker/Dockerfile -t <registry>/llmops-chat:latest .
docker push <registry>/llmops-chat:latest

# Edit k8s/chat-app/deployment.yaml → set image: to your pushed tag

./scripts/k8s-deploy.sh
```

The script applies manifests in dependency order:
RBAC → OTel Collector → Prometheus → Grafana → node-exporter → DCGM exporter → prometheus-adapter → Ollama (PVC + Deployment + Service) → chat-app → HPA.

The HPA is applied last because it requires prometheus-adapter to already be serving the external metrics API.

### Verify autoscaling is wired up

```bash
# Check the external metrics API is live
kubectl get --raw '/apis/external.metrics.k8s.io/v1beta1' | jq .

# Check the HPA sees the metric (may show <unknown> for ~60s after first deploy)
kubectl get hpa -n llmops

# Inspect current metric value
kubectl get --raw \
  '/apis/external.metrics.k8s.io/v1beta1/namespaces/llmops/llm_active_requests_total' \
  | jq .
```

### Hit the API

```bash
ENDPOINT=$(kubectl get svc chat-app -n llmops \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl -X POST http://$ENDPOINT/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"ministral-3:8b-instruct-2512-q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

Endpoints mirrored from Ollama: `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /health`, `GET /metrics`.

### Dashboards

Dashboard **LLMOps Multi-Replica Serving** is pre-loaded in Grafana.

Key panels beyond the single-node baseline:

| Panel | What it shows |
|-------|--------------|
| Ollama Replicas (Ready) | `kube_deployment_status_replicas_ready` |
| Active Requests vs Capacity | `llm:active_requests:total` vs `target × replicas` |
| HPA Scaling Events | Ready / Desired / Min / Max replica count over time |
| GPU util per node | Per-GPU `DCGM_FI_DEV_GPU_UTIL` labelled by `Hostname` |
| Little's Law vs gauge | Cross-validates the λW approximation against the direct gauge |

### Teardown

```bash
kubectl delete namespace llmops
kubectl delete apiservice v1beta1.external.metrics.k8s.io v1beta1.custom.metrics.k8s.io
```

---

## Configuration reference

All options read from `.env` (single-node) or the `chat-app-config` ConfigMap (K8s).

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `ministral-3:8b-instruct-2512-q4_K_M` | Model tag to load and serve |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTel Collector base URL |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | derived from base + `/v1/metrics` | Override only if collector splits signals |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | derived from base + `/v1/traces` | Override only if collector splits signals |
| `OTEL_SERVICE_NAME` | `llmops-chat` | Service label in traces and Grafana |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `dev` | Environment tag (`dev`, `kubernetes`, etc.) |

---

## VRAM sizing

VRAM estimate (Q4\_K\_M, context 60 000 tokens, batch 1):

```
VRAM = (P × b_w) + (0.55 + 0.08 × P) + (B × N × 2 × L × (d/g) × b_kv) × 10⁻⁹
     ≈ 9.4 GB
```

Where P = 8B params, b\_w = 0.5 B/param (Q4), B = 1, N = 60 000, L = 34 layers, d = 4096, g = 4 (GQA), b\_kv = 2.

Target provisioning: **12 GB VRAM** to leave headroom for runtime variability. Each Ollama replica in the multi-node deployment requires one GPU meeting this budget.
