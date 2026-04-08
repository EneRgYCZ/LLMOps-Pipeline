from fastapi import FastAPI
from pydantic import BaseModel
import requests
import time
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response, StreamingResponse
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
CONTEXT_LENGTH = Histogram("llm_context_length_tokens", "Total context length")

OLLAMA_URL = "http://ollama:11434/api/generate"

@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = req.prompt

    REQUESTS.inc()
    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "ministral-3:8b-instruct-2512-q4_K_M",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    def stream_generator():
        output_text = ""

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")

            output_text += token

            yield token

        latency = time.time() - start
        LATENCY.observe(latency)

        tokens_in = len(tokenizer.encode(prompt))
        tokens_out = len(tokenizer.encode(output_text))

        TOKENS_IN.inc(tokens_in)
        TOKENS_OUT.inc(tokens_out)

        if latency > 0:
            TOKENS_PER_SEC.set(tokens_out / latency)

        CONTEXT_LENGTH.observe(tokens_in + tokens_out)

    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")