import asyncio
import logging
import os

# ---------------------------------------------------------------------------
# Verbose HTTP logging — show the exact request/response going to Ollama
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("openai._base_client").setLevel(logging.DEBUG)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M")
N_CALLS = int(os.getenv("N_CALLS", "3"))


async def main():
    from openai import AsyncOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    print("=" * 70)
    print("RQ3 DIAGNOSTIC")
    print(f"OLLAMA_HOST={OLLAMA_HOST}  OLLAMA_MODEL={OLLAMA_MODEL}  N_CALLS={N_CALLS}")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 1: inspect what params Faithfulness / the metric base class accept
    # ---------------------------------------------------------------------
    print("\n--- Step 1: Faithfulness signature inspection ---")
    import inspect

    print("Faithfulness.__init__ signature:")
    print(" ", inspect.signature(Faithfulness.__init__))
    print("Faithfulness.ascore signature:")
    print(" ", inspect.signature(Faithfulness.ascore))

    # Look for any internal generation config (temperature, top_p, etc.)
    print("\nFaithfulness instance attributes (after construction):")

    # ---------------------------------------------------------------------
    # Step 2: build the judge LLM exactly as in rq3_experiment.py
    # ---------------------------------------------------------------------
    client = AsyncOpenAI(api_key="ollama", base_url=f"{OLLAMA_HOST}/v1")
    llm = llm_factory(
        OLLAMA_MODEL,
        provider="openai",
        client=client,
        max_tokens=4096,
        temperature=0.1,
    )
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    faithfulness = Faithfulness(llm=llm)

    for attr in ["temperature", "llm_config", "generation_config", "model_config"]:
        if hasattr(faithfulness, attr):
            print(f"  faithfulness.{attr} = {getattr(faithfulness, attr)}")
    if hasattr(llm, "temperature"):
        print(f"  llm.temperature = {llm.temperature}")
    for attr in dir(llm):
        if (
            "temp" in attr.lower()
            or "sampl" in attr.lower()
            or "config" in attr.lower()
        ):
            try:
                print(f"  llm.{attr} = {getattr(llm, attr)}")
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Step 3: one fixed sample, run N_CALLS times, same process
    # ---------------------------------------------------------------------
    print("\n--- Step 2: fixed sample scored 3x in same process (no restarts) ---")
    sample = SingleTurnSample(
        user_input="What are the global implications of the USA Supreme Court ruling on abortion?",
        response=(
            "The global implications of the USA Supreme Court ruling on abortion can be "
            "significant, as it sets a precedent for other countries and influences global "
            "discourse on reproductive rights. Different countries may respond differently, "
            "with some strengthening protections for abortion rights and others using it to "
            "justify restrictions."
        ),
        retrieved_contexts=[
            "The USA Supreme Court ruled in 2022 to overturn Roe v. Wade, eliminating the "
            "federal constitutional right to abortion and returning regulation to individual states."
        ],
        reference=(
            "The ruling has prompted varied international reactions, with human rights "
            "organisations expressing concern over potential global rollback of reproductive rights."
        ),
    )

    scores = []
    for i in range(1, N_CALLS + 1):
        print(f"\n  --- call {i}/{N_CALLS} ---")
        try:
            result = await faithfulness.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
            )
            score = result.value if hasattr(result, "value") else result
            scores.append(float(score))
            print(f"  -> score = {score}")

            # Try to surface any intermediate reasoning/statements if exposed
            for attr in [
                "reason",
                "statements",
                "verdicts",
                "raw_output",
                "_last_response",
            ]:
                if hasattr(result, attr):
                    print(f"  -> result.{attr} = {getattr(result, attr)}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            scores.append(None)

    print("\n--- Summary ---")
    print(f"  Scores across {N_CALLS} calls: {scores}")
    if len(set(s for s in scores if s is not None)) == 1:
        print("  -> IDENTICAL scores across all calls.")
    else:
        print("  -> Scores DIFFER across calls (this is what we want to see).")

    # ---------------------------------------------------------------------
    # Step 4: raw chat completion, bypassing RAGAS entirely
    # ---------------------------------------------------------------------
    print("\n--- Step 3: raw completion via the SAME client, bypassing RAGAS ---")
    raw_outputs = []
    for i in range(1, N_CALLS + 1):
        resp = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Write one random short sentence about the weather.",
                }
            ],
            temperature=0.1,
            max_tokens=50,
        )
        text = resp.choices[0].message.content
        raw_outputs.append(text)
        print(f"  call {i}: {text!r}")

    print("\n--- Raw completion comparison ---")
    if len(set(raw_outputs)) == 1:
        print(
            "  -> Raw completions are IDENTICAL even with temperature=0.1 and a "
            "loosely-specified prompt."
        )
        print(
            "  -> This points to something at the Ollama/server level pinning "
            "generation (caching, fixed seed, or temperature not applied)."
        )
    else:
        print(
            "  -> Raw completions DIFFER across calls — temperature IS working "
            "at the raw completion level."
        )
        print(
            "  -> If RAGAS scores were still identical in Step 2, the issue is "
            "in RAGAS's structured-output parsing converging on the same score "
            "despite varied wording (expected behaviour, not a bug)."
        )

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE — paste this full output back for interpretation.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
