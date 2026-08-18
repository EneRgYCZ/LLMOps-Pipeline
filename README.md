# LLMOps Pipeline

End-to-end LLM inference pipeline with two deployment strategies and a shared observability stack, built for a UvA Master's thesis on LLMOps. Retrieval-augmented generation is the use case driving the pipeline throughout.

Default model: **Ministral 3 8B Instruct Q4_K_M** (~9.4 GB VRAM). Inference via [Ollama](https://ollama.com). Monitoring via Prometheus + Grafana + OpenLIT (OpenTelemetry). RAG via ChromaDB. Evaluation via RAGAS. Runs on CPU-only or NVIDIA GPU machines through a single `TARGET` switch in `.env`.

## Contents

- [Repository layout](#repository-layout)
- [Strategy 1 — single-node (Docker Compose)](#strategy-1--single-node-docker-compose)
- [Strategy 2 — Kubernetes multi-replica](#strategy-2--kubernetes-multi-replica)
- [Configuration](#configuration)
- [Tests](#tests)
- [Thesis experiments](#thesis-experiments)
- [VRAM sizing](#vram-sizing)
- [License](#license)

## Repository layout

```
single-node/    Strategy 1: Docker Compose (CPU or GPU overlay), Prometheus, Grafana, OTel Collector
k8s/            Strategy 2: Kubernetes manifests (Ollama, chat-app, ChromaDB, monitoring, HPA)
src/            Application code: chat client, FastAPI wrapper, RAG, evaluation, telemetry, settings
docker/         Dockerfile for the chat-app image used in k8s
scripts/        start.sh, k8s-deploy.sh, ingest.py, run-tests.sh, load_references.py
tests/          pytest suite (no external services required; all calls mocked)
analysis/       Thesis experiment scripts (RQ3 evaluation reliability, RQ4 VRAM formula validation)
results/        Raw data, CSVs, and figures produced by the RQ3/RQ4 experiments
corpus/         Sample documents ingested into ChromaDB for RAG
refs/           Reference query/answer pairs used to enable RAGAS context_recall
```

## How it works

A shared observability stack instruments both deployment strategies: OpenLIT captures LLM-level metrics (time-to-first-token, tokens/s, latency, errors) via OTel, while node-exporter (and DCGM in GPU mode) covers hardware. Both feed Prometheus and a pre-built Grafana dashboard, so a latency spike can be traced to GPU saturation, thermal throttling, or context-length growth.

When `RAG_ENABLED=true`, prompts are augmented with context retrieved from ChromaDB (`scripts/ingest.py` loads documents). When `EVAL_ENABLED=true`, a background asyncio worker scores each response with RAGAS (faithfulness, answer relevancy, context recall) without adding to request latency, storing results in SQLite.

## Strategy 1 — single-node (Docker Compose)

Single Ollama container, one request at a time. Simplest topology; used as the observability baseline.

**Prerequisites:** Linux, Docker Compose v2. GPU mode additionally needs an NVIDIA GPU (≥12 GB VRAM recommended), driver installed, and the NVIDIA Container Toolkit (installed automatically by `start.sh` if missing).

```bash
cp .env.example .env          # set TARGET=CPU or TARGET=GPU
./scripts/start.sh            # starts the stack and pulls the model

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/chat.py            # chat client
```

To enable RAG: `python scripts/ingest.py --chroma-host localhost --chroma-port 8100 --collection documents --input-dir ./corpus/`, then set `RAG_ENABLED=true` and restart.

Dashboards: `http://localhost:3000` (admin/admin), **LLMOps** dashboard pre-loaded.

Stop: `docker compose -f single-node/docker-compose.yml [-f single-node/docker-compose.gpu.yml] down`

## Strategy 2 — Kubernetes multi-replica

Each Ollama pod holds a full model copy; a Service round-robins requests across pods, and an HPA scales replica count on in-flight request count (`target: AverageValue 3`, sourced from a Prometheus recording rule via `prometheus-adapter`).

**Prerequisites:** a cluster with `kube-state-metrics`, `kubectl`, Docker. GPU mode needs a ReadWriteMany StorageClass (Ollama replicas share a PVC), GPU nodes with the NVIDIA device plugin, and `nvidia-container-runtime`/GPU Operator.

```bash
cp .env.example .env
docker build -f docker/Dockerfile -t <registry>/llmops-chat:latest .
docker push <registry>/llmops-chat:latest
# set image: in k8s/chat-app/deployment.yaml to the pushed tag

./scripts/k8s-deploy.sh
```

> The committed manifest templates contain `${PLACEHOLDER}` variables and are not valid YAML until rendered — always deploy via `k8s-deploy.sh`, never `kubectl apply -f k8s/...` directly.

```bash
kubectl get hpa -n llmops
ENDPOINT=$(kubectl get svc chat-app -n llmops -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$ENDPOINT/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"ministral-3:8b-instruct-2512-q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

Endpoints mirror Ollama: `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /health`, `GET /metrics`. Dashboard **LLMOps Multi-Replica Serving** is pre-loaded.

**Known limitation:** in CPU mode, each Ollama replica uses `emptyDir` (no ReadWriteMany requirement) and pulls the model fresh on startup, so scale-out incurs a full model pull per replica.

Teardown: `kubectl delete namespace llmops && kubectl delete apiservice v1beta1.external.metrics.k8s.io v1beta1.custom.metrics.k8s.io`

## Configuration

All options are read from `.env` (single-node) or the `chat-app-config` ConfigMap (k8s). See `.env.example` for the full list with defaults; key groups:

| Group | Variables |
|-------|-----------|
| Inference | `TARGET`, `OLLAMA_HOST`, `OLLAMA_MODEL` |
| RAG | `RAG_ENABLED`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION` |
| Evaluation | `EVAL_ENABLED`, `EVAL_DB_PATH`, `EVAL_SAMPLE_RATE`, `EVAL_JUDGE_TEMPERATURE`, `EVAL_JUDGE_TOP_P`, `EVAL_EMBEDDING_MODEL`, `EVAL_REFERENCES_PATH` |
| Observability | `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `OTEL_DEPLOYMENT_ENVIRONMENT`, `OTEL_METRICS_EXPORTER`, `OTEL_TRACES_EXPORTER` |

`TARGET` (`CPU`/`GPU`, case-insensitive, default `CPU`) toggles GPU resource requests, `runtimeClassName`, the DCGM exporter, PVC vs `emptyDir` model storage, and startup probe timeouts — see `scripts/start.sh` and `scripts/k8s-deploy.sh` for the exact conditionals.

## Tests

```bash
pip install -r requirements-test.txt
./scripts/run-tests.sh
```

No external services required — all external calls are mocked.

## Thesis experiments

`analysis/` holds the standalone experiment scripts backing the thesis's empirical results (raw output in `results/`); folder names are internal experiment identifiers, not the thesis's RQ numbering:

- **`analysis/rq3/`** — repeated-measures reliability of the pipeline's own RAGAS evaluation metrics (ICC(2,1) across runs), plus a Jupyter notebook investigating a faithfulness drift effect. Backs the thesis's RQ3 (evaluation reliability).
- **`analysis/rq4/`** — validates the analytical VRAM estimation formula (see below) against measured GPU telemetry across model quantisation levels. Backs the thesis's RQ2 (resource/hardware requirements).

These scripts run standalone against a live Ollama instance and are decoupled from the serving infrastructure above.

## VRAM sizing

```
VRAM = (P × b_w) + (0.55 + 0.08 × P) + (B × N × 2 × L × (d/g) × b_kv) × 10⁻⁹  ≈  9.4 GB
```

For the default model: P = 8B params, b_w = 0.5 B/param (Q4), B = 1, N = 60,000 context tokens, L = 34 layers, d = 4096, g = 4 (GQA), b_kv = 2. Target provisioning: **12 GB VRAM** for headroom; each Kubernetes Ollama replica needs one GPU meeting this budget.

## License

[Apache 2.0](LICENSE)
