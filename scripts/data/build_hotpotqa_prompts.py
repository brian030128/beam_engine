"""Build a JSONL of long-shared-prefix prompts from HotpotQA.

Each output line mirrors ``data/gov_report.jsonl``:
    {"token_len": int, "prompt": str}

The prompt structure mimics the multi-document RAG setting we want to test
with the K=16 B=1 L_p≈40k cell from v11:

    Document 1: <title>
    <paragraph sentences>

    Document 2: ...

    ...

    Document N: ...

    Question: <one HotpotQA question>
    Answer:

We concatenate context paragraphs from many distinct HotpotQA examples
until the total token count exceeds the target L_p, then append one
final question. Beam search at K=16 generates K=16 candidate answer
completions over the same shared context.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Source dataset: hotpotqa/hotpot_qa (multi-hop QA, ~90k train examples).
# "distractor" config has 10 paragraphs per example (some supporting, some
# distractors). We pull context paragraphs from many examples to build the
# long shared prefix and pick one example's question as the final query.
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"  # smaller (7.4k examples), good enough for source pool

DEFAULT_OUT = Path("data/hotpotqa.jsonl")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_TARGET_TOKENS = 45000  # safely above L_p=40000
DEFAULT_N_PROMPTS = 30


def build_prompt(rng: random.Random, examples, tokenizer, target_tokens: int) -> tuple[str, int]:
    """Pick a question + concatenate distinct documents until token count ≥ target."""
    # Pick the "query" example: its question goes at the end.
    query_idx = rng.randrange(len(examples))
    query_ex = examples[query_idx]

    # Shuffle pool of OTHER examples to draw context paragraphs from.
    other_indices = [i for i in range(len(examples)) if i != query_idx]
    rng.shuffle(other_indices)

    parts: list[str] = []
    seen_titles: set[str] = set()
    doc_no = 0

    # First, include the supporting context from the query example so the
    # question is at least answerable in principle.
    qctx = query_ex["context"]
    for title, sents in zip(qctx["title"], qctx["sentences"]):
        if title in seen_titles:
            continue
        seen_titles.add(title)
        doc_no += 1
        body = " ".join(sents).strip()
        if not body:
            continue
        parts.append(f"Document {doc_no}: {title}\n{body}\n")

    # Add distractor paragraphs from other examples until target.
    suffix = f"\nQuestion: {query_ex['question']}\nAnswer:"

    def total_tok_len(parts_list: list[str]) -> int:
        body = "".join(parts_list) + suffix
        # add_special_tokens=False so we get a clean prefix count; bench
        # will add BOS itself.
        return len(tokenizer(body, add_special_tokens=False).input_ids)

    for idx in other_indices:
        ex = examples[idx]
        ctx = ex["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            doc_no += 1
            body = " ".join(sents).strip()
            if not body:
                continue
            parts.append(f"Document {doc_no}: {title}\n{body}\n")
        tl = total_tok_len(parts)
        if tl >= target_tokens:
            break

    prompt = "".join(parts) + suffix
    tl = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return prompt, tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer model id (for accurate token counts)")
    ap.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/work/u4320956/hf"))
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    # use cached HF token if present
    tok_file = os.path.expanduser("~/.cache/huggingface/token")
    if not os.environ.get("HF_TOKEN") and os.path.exists(tok_file):
        with open(tok_file) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading dataset {DATASET}:{CONFIG} split={SPLIT} ...", file=sys.stderr)
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, trust_remote_code=False)
    examples = list(ds)
    print(f"  loaded {len(examples)} examples", file=sys.stderr)

    print(f"Loading tokenizer {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w") as f:
        for i in range(args.n_prompts):
            prompt, tl = build_prompt(rng, examples, tokenizer, args.target_tokens)
            f.write(json.dumps({"token_len": tl, "prompt": prompt}) + "\n")
            written += 1
            print(f"  [{i+1}/{args.n_prompts}] token_len={tl}", file=sys.stderr)

    print(f"\nWrote {written} prompts to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
