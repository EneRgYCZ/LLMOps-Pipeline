import ollama

from settings import Settings


class ChatApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = ollama.Client(host=settings.chat_host)
        self.messages: list[dict[str, str]] = []

    def run(self) -> None:
        print("LLM Chat (type 'exit' to quit)\n")
        while True:
            try:
                prompt = input("You: ").strip()

                if prompt.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if not prompt:
                    continue

                self.messages.append({"role": "user", "content": prompt})
                print("\nAssistant:\n")
                assistant_text = self._stream_response()
                self.messages.append({"role": "assistant", "content": assistant_text})
                print("\n" + "-" * 60 + "\n")

            except ollama.ResponseError as exc:
                print(f"\nOllama error: {exc.error} (status {exc.status_code})")
                print("Is the model loaded?  Run: ollama pull", self.settings.ollama_model)
                self.messages.pop()  # discard the unanswered user turn
            except KeyboardInterrupt:
                print("\nInterrupted. Bye!")
                break

    def _stream_response(self) -> str:
        response_stream = self.client.chat(
            model=self.settings.ollama_model,
            messages=self.messages,
            stream=True,
        )

        chunks: list[str] = []
        for chunk in response_stream:
            # ollama SDK 0.4.x returns ChatResponse objects; .message.content is the
            # delta text for this chunk (empty string on the final done=True chunk).
            content = chunk.message.content or ""
            if content:
                chunks.append(content)
                print(content, end="", flush=True)

        return "".join(chunks)
