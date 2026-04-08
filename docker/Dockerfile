FROM python:3.11-slim

WORKDIR /app

# Avoid Python buffering (important for logs)
ENV PYTHONUNBUFFERED=1

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY src/llm_wrapper.py .

# Run
CMD ["uvicorn", "llm_wrapper:app", "--host", "0.0.0.0", "--port", "8000"]