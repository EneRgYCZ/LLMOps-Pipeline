from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
from transformers import AutoTokenizer

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

class GenerateRequest(BaseModel):
    prompt: str

REQUESTS = Counter("llm_requests_total", "Total LLM requests")
LATENCY = Histogram("llm_latency_seconds", "LLM request latency")
TOKENS_OUT = Counter("llm_output_tokens_total", "Generated tokens")
TOKENS_IN = Counter("llm_input_tokens_total", "Input tokens")
TOKENS_PER_SEC = Gauge("llm_tokens_per_second", "Throughput")
CONTEXT_LENGTH = Histogram("llm_context_length_tokens", "Total context length (input + output)")

OLLAMA_URL = "http://ollama:11434/api/generate"

@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = req.prompt

    start = time.time()
    REQUESTS.inc()

    response = requests.post(OLLAMA_URL, json={
        "model": "ministral-3:8b-instruct-2512-q4_K_M",
        "prompt": prompt,
        "stream": False
    })

    data = response.json()

    latency = time.time() - start
    LATENCY.observe(latency)

    output_text = data.get("response", "")

    tokens_in = len(tokenizer.encode(prompt))
    tokens_out = len(tokenizer.encode(output_text))

    TOKENS_IN.inc(tokens_in)
    TOKENS_OUT.inc(tokens_out)

    if latency > 0:
        TOKENS_PER_SEC.set(tokens_out / latency)

    context_len = tokens_in + tokens_out
    CONTEXT_LENGTH.observe(context_len)

    return data

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")