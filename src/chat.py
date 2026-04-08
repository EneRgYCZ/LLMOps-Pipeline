import os

import ollama
import openlit

# Configure OpenLIT via env vars (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, etc.)
openlit.init()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M")


def main() -> None:
    client = ollama.Client(host=OLLAMA_HOST)
    messages = []

    print("LLM Chat with OpenLIT monitoring (type 'exit' to quit)\n")

    while True:
        try:
            prompt = input("You: ").strip()

            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not prompt:
                continue

            messages.append({"role": "user", "content": prompt})

            print("\nAssistant:\n")

            response_stream = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
            )

            assistant_chunks = []
            for chunk in response_stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    assistant_chunks.append(content)
                    print(content, end="", flush=True)

            assistant_text = "".join(assistant_chunks)
            messages.append({"role": "assistant", "content": assistant_text})

            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            break


if __name__ == "__main__":
    main()