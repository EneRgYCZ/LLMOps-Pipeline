import asyncio
import subprocess
import time
from rq3_experiment import build_ragas_components, build_metrics, load_amnesty_qa

async def main():
    llm, embeddings = build_ragas_components()
    metrics = build_metrics(llm, embeddings)
    sample = load_amnesty_qa()[0]

    for i in range(5):
        print(f"### RUN {i + 1} ###")
        if i > 0:
            subprocess.run(
                ["docker", "exec", "ollama", "ollama", "stop",
                 "ministral-3:8b-instruct-2512-q4_K_M"],
                timeout=30,
            )
            time.sleep(2)
        score = await metrics["faithfulness"].ascore(
            user_input=sample["question"],
            response=sample["answer"],
            retrieved_contexts=sample["contexts"],
        )
        print("SCORE:", score.value)

asyncio.run(main())
