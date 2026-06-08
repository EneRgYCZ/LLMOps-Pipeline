FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch before requirements.txt so pip does not pull the CUDA
# variant. libc10_cuda.so's global constructor crashes in containerized builds
# where the kernel clock is non-monotonic under seccomp; the chat-app never needs GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ .

CMD ["uvicorn", "llm_wrapper:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
