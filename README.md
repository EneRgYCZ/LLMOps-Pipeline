# LLMOps Pipeline

End-to-end LLM inference pipeline with two deployment strategies and a shared observability stack. Built for the UvA Master's thesis on LLMOps.

Default model: **Ministral 3 8B Instruct Q4\_K\_M** (~9.4 GB VRAM). Inference via [Ollama](https://ollama.com). Monitoring via Prometheus + Grafana + OpenLIT (OpenTelemetry). Supports CPU-only and NVIDIA GPU machines via a single `TARGET` switch in `.env`.

---

## Repository layout

```
.
├── single-node/                  # Strategy 1: Docker Compose (CPU or GPU)
│   ├── docker-compose.yml        # Base stack (CPU resource limits)
│   ├── docker-compose.gpu.yml    # GPU overlay (loaded automatically when TARGET=GPU)
│   ├── otelcol.yaml              # OpenTelemetry Collector config
│   ├── prometheus.yml            # Scrape config (15s interval)
│   ├── prometheus-rules.yml      # Recording rules (latency, queue depth, error rate)
│   └── grafana/
│       ├── provisioning/         # Auto-provisioned datasource + dashboard provider
│       └── dashboards/           # Pre-built Grafana dashboard JSON
│
├── k8s/                          # Strategy 2: Kubernetes multi-replica + HPA
│   ├── namespace.yaml
│   ├── ollama/                   # Deployment (envsubst template), Service, PVC, HPA
│   ├── chat-app/                 # FastAPI wrapper Deployment + ConfigMap (envsubst template) + Service
│   ├── chromadb/                 # ChromaDB StatefulSet + Service
│   ├── monitoring/               # Prometheus (envsubst template), Grafana, OTel, node-exporter, DCGM
│   └── prometheus-adapter/       # Bridges Prometheus → K8s External Metrics API for HPA
│
├── src/
│   ├── chat.py                   # CLI entry point
│   ├── chat_app.py               # ChatApplication class (streaming, history)
│   ├── evaluation.py             # RAGAS evaluation (background asyncio worker; never blocks inference)
│   ├── llm_wrapper.py            # FastAPI proxy (used in K8s; exposes /metrics)
│   ├── rag.py                    # RAG module (ChromaDB retrieval, prompt augmentation)
│   ├── settings.py               # Settings dataclass + env loader
│   └── telemetry.py              # OpenTelemetry + OpenLIT initialisation
│
├── docker/
│   └── Dockerfile                # Image for llm_wrapper.py (used in K8s chat-app)
│
├── scripts/
│   ├── start.sh                  # Single-node: start stack, pull model (reads TARGET from .env)
│   ├── k8s-deploy.sh             # Kubernetes: render templates, build image, apply manifests
│   ├── ingest.py                 # Ingest documents into ChromaDB for RAG
│   ├── load_references.py        # Manage evaluation reference datasets (ground-truth answers)
│   └── run-tests.sh              # Run test suite locally (no external services required)
│
├── tests/                        # pytest test suite
├── requirements.txt
├── requirements-test.txt
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
                              │  dcgm-exporter  (GPU, VRAM) │  ← TARGET=GPU only
                              └─────────────┬──────────────┘
                                            │
                                        Grafana
```

Two monitoring tiers:
- **Hardware layer** — node-exporter (CPU, RAM, disk) always present. DCGM exporter (GPU utilisation, VRAM, temperature) added when `TARGET=GPU`.
- **LLM inference layer** — OpenLIT instruments the Ollama client calls and emits OTel metrics: time-to-first-token, tokens/s, operation latency percentiles, error rate.

A latency spike in the inference layer can be cross-correlated with the hardware layer to determine whether the cause is GPU saturation, thermal throttling, or context-length growth.

### RAG and evaluation

When `RAG_ENABLED=true`, the pipeline retrieves context from ChromaDB before sending prompts to Ollama. Documents are ingested via `scripts/ingest.py`.

When `EVAL_ENABLED=true`, the `evaluation.py` module runs RAGAS quality metrics (faithfulness, answer relevancy, context recall) in a background asyncio worker after each response. Results are stored in a SQLite database (`EVAL_DB_PATH`). Inference latency is never affected.

---

## Strategy 1 — Single-node (Docker Compose)

```
Client (terminal)
    └── Ollama  (single container, CPU or GPU)
          └── model loaded in RAM/VRAM
```

Ollama handles queuing. One request at a time. Simplest topology; used as the observability baseline.

### Prerequisites

**Both modes:**
- Linux host
- Docker + Docker Compose v2

**GPU mode only (`TARGET=GPU`):**
- NVIDIA GPU (≥12 GB VRAM recommended for the default model)
- NVIDIA driver installed (`nvidia-smi` works)
- NVIDIA Container Toolkit (installed automatically by `start.sh` if absent)

### Start

```bash
cp .env.example .env   # set TARGET=CPU or TARGET=GPU, edit OLLAMA_MODEL if needed
./scripts/start.sh
```

`start.sh` reads `TARGET` from `.env` and:
1. **GPU mode only:** Installs NVIDIA Container Toolkit if absent (requires sudo)
2. Starts the stack — base `docker-compose.yml` plus `docker-compose.gpu.yml` overlay in GPU mode
3. Pulls the inference model and embedding model into Ollama

### Run the chat client

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/chat.py
```

The client reads `OLLAMA_HOST` and `OTEL_EXPORTER_OTLP_ENDPOINT` from `.env` automatically.

### Ingest documents (RAG)

```bash
python scripts/ingest.py --chroma-host localhost --chroma-port 8100 \
    --collection documents --input-dir ./corpus/
```

Then set `RAG_ENABLED=true` in `.env` and restart the stack.

### Dashboards

Open Grafana at `http://localhost:3000` (admin / admin). The **LLMOps** dashboard is pre-loaded.

| Panel | Metric source |
|-------|--------------|
| Request rate / error rate | `llm_requests_total` |
| Latency p50 / p95 / p99 | `llm_request_duration_seconds` histogram |
| OTel operation latency | `gen_ai_client_operation_duration_seconds` (OpenLIT) |
| GPU utilisation | `DCGM_FI_DEV_GPU_UTIL` (GPU mode only) |
| GPU VRAM used | `DCGM_FI_DEV_FB_USED` (GPU mode only) |
| Host CPU / RAM | `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes` |
| Little's Law queue depth | recording rule: `λ × W` |

### Stop

```bash
# CPU mode
docker compose -f single-node/docker-compose.yml down

# GPU mode (include overlay so dcgm-exporter is also stopped)
docker compose -f single-node/docker-compose.yml -f single-node/docker-compose.gpu.yml down
```

---

## Strategy 2 — Kubernetes multi-replica serving

```
External client
    └── LoadBalancer Service
          └── chat-app pods (FastAPI, llm_wrapper.py)
                └── ollama Service (round-robin ClusterIP)
                      ├── ollama pod 0  (node 0)
                      ├── ollama pod 1  (node 1)
                      └── ollama pod N  (node N)
```

Each Ollama pod holds one full copy of the model. The Kubernetes Service distributes requests across pods (no session affinity). The Horizontal Pod Autoscaler adds or removes Ollama pods based on in-flight request count.

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

**Both modes:**
- Kubernetes cluster with `kube-state-metrics` running in `kube-system` (standard in most distributions)
- `kubectl` configured for the target cluster
- Docker (to build the chat-app image)

**GPU mode only (`TARGET=GPU`):**
- StorageClass supporting **ReadWriteMany** (NFS, CephFS, AWS EFS, etc.) — Ollama replicas share a single PVC
- NVIDIA GPU nodes with NVIDIA device plugin (`nvidia.com/gpu` resource)
- `nvidia-container-runtime` or GPU Operator installed on each node

**CPU mode (`TARGET=CPU`):**
- No ReadWriteMany StorageClass required — each Ollama replica uses `emptyDir` and pulls its own model copy on startup

### Deploy

```bash
cp .env.example .env   # set TARGET=CPU or TARGET=GPU, set OLLAMA_MODEL

# Build and push image (replace registry prefix as needed)
docker build -f docker/Dockerfile -t <registry>/llmops-chat:latest .
docker push <registry>/llmops-chat:latest

# Edit k8s/chat-app/deployment.yaml → set image: to your pushed tag

./scripts/k8s-deploy.sh
```

The script reads `TARGET` from `.env`, renders the hardware-sensitive manifests via `envsubst`, then applies everything in dependency order:

RBAC → OTel Collector → Prometheus → Grafana → node-exporter → DCGM exporter *(GPU mode only)* → prometheus-adapter → Ollama (PVC + Deployment + Service) → chat-app → HPA.

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
| GPU util per node | Per-GPU `DCGM_FI_DEV_GPU_UTIL` labelled by `Hostname` (GPU mode only) |
| Little's Law vs gauge | Cross-validates the λW approximation against the direct gauge |

### Teardown

```bash
kubectl delete namespace llmops
kubectl delete apiservice v1beta1.external.metrics.k8s.io v1beta1.custom.metrics.k8s.io
```

---

## Hardware target (`TARGET`)

Set `TARGET` in `.env` to select the hardware mode. Accepted values: `GPU` or `CPU` (case-insensitive). Default: `CPU`.

```env
TARGET=CPU   # CPU-only machine
TARGET=GPU   # NVIDIA GPU machine
```

`OLLAMA_MODEL` is independent of `TARGET` — set it separately to any model tag Ollama supports.

### What TARGET toggles

| Feature | `GPU` | `CPU` |
|---------|-------|-------|
| Ollama resource requests/limits | `nvidia.com/gpu: "1"` | cpu 1–3 / memory 6–12 Gi |
| `runtimeClassName` (K8s only) | `nvidia` | absent |
| `NVIDIA_VISIBLE_DEVICES` env var | `all` | absent |
| DCGM exporter | deployed | skipped |
| Grafana GPU panels | data from DCGM | "No data" (panel stays) |
| K8s startup probe window | 10 min (60 × 10 s) | 20 min (120 × 10 s) |
| K8s Ollama model storage | shared PVC (ReadWriteMany) | `emptyDir` per replica |
| K8s PVC applied | yes | skipped |
| Docker Compose GPU device reservation | present | absent |

### Single-node (Docker Compose)

`./scripts/start.sh` reads `TARGET` from `.env`. In GPU mode it appends
`single-node/docker-compose.gpu.yml` as a Compose overlay, which adds the
NVIDIA device reservation to the Ollama service and starts the DCGM exporter.
In CPU mode only the base `docker-compose.yml` is used.

### Kubernetes

`./scripts/k8s-deploy.sh` reads `TARGET` from `.env`, sets template variables,
renders the manifests listed below via `envsubst`, then runs `kubectl apply`:

- `k8s/ollama/deployment.yaml` — GPU/CPU resources, runtime class, env, volume, startup probe
- `k8s/chat-app/configmap.yaml` — model tag (`OLLAMA_MODEL`)
- `k8s/monitoring/prometheus/configmap.yaml` — GPU scrape job

The script also applies whole resources conditionally:

| Resource | `GPU` | `CPU` |
|----------|-------|-------|
| `k8s/ollama/pvc.yaml` | applied | **skipped** |
| `k8s/monitoring/dcgm-exporter/` | applied | **skipped** |

> **Important:** The committed versions of the template files above contain
> `${PLACEHOLDER}` variables and cannot be applied directly with `kubectl apply -f`.
> Running `kubectl apply -f k8s/...` on the raw files will produce invalid YAML.
> Always deploy via `./scripts/k8s-deploy.sh`.

**Known limitation — CPU mode model cold-start:** In CPU mode each Ollama replica
uses an `emptyDir` volume (no ReadWriteMany StorageClass required). This means
every new pod pulls the model from the Ollama registry on startup. Scale-out events
therefore incur a full model pull per replica in addition to the normal load time;
the startup probe's 20-minute window accounts for this.

---

## Configuration reference

All options read from `.env` (single-node) or the `chat-app-config` ConfigMap (K8s).

### Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET` | `CPU` | Hardware mode: `GPU` or `CPU` (case-insensitive) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `ministral-3:8b-instruct-2512-q4_K_M` | Model tag to load and serve |

### RAG

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLED` | `false` | Enable retrieval-augmented generation |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8100` | ChromaDB port |
| `CHROMA_COLLECTION` | `documents` | ChromaDB collection name |

### Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_ENABLED` | `true` | Enable background RAGAS evaluation |
| `EVAL_DB_PATH` | `/data/evaluations.db` | SQLite path for storing evaluation results |
| `EVAL_SAMPLE_RATE` | `1.0` | Fraction of requests to evaluate (0.0–1.0) |
| `EVAL_JUDGE_TEMPERATURE` | `0.1` | RAGAS judge LLM sampling temperature |
| `EVAL_JUDGE_TOP_P` | `0.95` | RAGAS judge LLM nucleus sampling top_p (must be set explicitly alongside temperature) |
| `EVAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace model used for RAGAS AnswerRelevancy embeddings |
| `EVAL_TIMEOUT_SECONDS` | `300.0` | Max seconds to wait for RAGAS evaluation |
| `EVAL_REFERENCES_PATH` | unset | JSON file of query/reference pairs enabling context_recall |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTel Collector base URL |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | derived | Override only if collector splits signals |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | derived | Override only if collector splits signals |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Wire protocol for OTel export |
| `OTEL_SERVICE_NAME` | `llmops-chat` | Service label in traces and Grafana |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `dev` | Environment tag (`dev`, `kubernetes`, etc.) |
| `OTEL_METRICS_EXPORTER` | `otlp` | Metrics exporter backend |
| `OTEL_TRACES_EXPORTER` | `otlp` | Traces exporter backend |
| `OTEL_LOGS_EXPORTER` | `none` | Logs exporter backend |

---

## Tests

```bash
pip install -r requirements-test.txt
./scripts/run-tests.sh
```

No external services required — all external calls are mocked in the test suite.

---

## VRAM sizing

VRAM estimate (Q4\_K\_M, context 60 000 tokens, batch 1):

```
VRAM = (P × b_w) + (0.55 + 0.08 × P) + (B × N × 2 × L × (d/g) × b_kv) × 10⁻⁹
     ≈ 9.4 GB
```

Where P = 8B params, b\_w = 0.5 B/param (Q4), B = 1, N = 60 000, L = 34 layers, d = 4096, g = 4 (GQA), b\_kv = 2.

Target provisioning: **12 GB VRAM** to leave headroom for runtime variability. Each Ollama replica in the multi-node deployment requires one GPU meeting this budget.
