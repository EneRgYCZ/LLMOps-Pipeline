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

        response = requests.post(URL, json={"prompt": prompt})

        if response.status_code != 200:
            print("⚠️ Error:", response.text)
            continue

        data = response.json()
        answer = data.get("response", "")

        print("\nAssistant:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Bye!")
        break