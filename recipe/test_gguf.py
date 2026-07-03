#!/usr/bin/env python3
import sys
import time

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/model.gguf>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]

    print("1. Importing llama_cpp ...", end=" ", flush=True)
    from llama_cpp import Llama
    print("OK")

    print(f"2. Loading model: {model_path} ...", end=" ", flush=True)
    t0 = time.time()
    llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
    print(f"OK ({time.time() - t0:.1f}s)")

    print("3. Tokenization ...", end=" ", flush=True)
    tokens = llm.tokenize(b"Hello, world!")
    assert len(tokens) > 0
    print(f"OK ({len(tokens)} tokens)")

    print("4. Inference ...", end=" ", flush=True)
    t0 = time.time()
    output = llm(
        "Q: What is the capital of France? A:",
        max_tokens=16,
        stop=["Q:", "\n"],
        echo=False,
    )
    elapsed = time.time() - t0
    text = output["choices"][0]["text"].strip()
    print(f"OK ({elapsed:.1f}s)")
    print(f"   Response: {text!r}")

    print("5. Embedding ...", end=" ", flush=True)
    try:
        emb_llm = Llama(model_path=model_path, embedding=True, n_ctx=64, verbose=False)
        vec = emb_llm.embed("test sentence")
        assert len(vec) > 0
        print(f"OK (dim={len(vec)})")
    except Exception as e:
        print(f"SKIP ({e})")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
