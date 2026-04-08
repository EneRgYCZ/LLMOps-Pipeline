import requests

URL = "http://localhost:8000/generate"

print("🧠 LLM Chat (type 'exit' to quit)\n")

while True:
    try:
        prompt = input("You: ").strip()

        if prompt.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if not prompt:
            continue

        response = requests.post(URL, json={"prompt": prompt}, stream=True)

        print("\nAssistant:\n")

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode("utf-8"), end="", flush=True)

        print("\n" + "-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Bye!")
        break