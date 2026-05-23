"""Build a JSONL of long-shared-prefix multi-question prompts from HotpotQA.

Each output line:
    {
      "prefix":     str,            # concatenated docs, ~40k+ tokens
      "questions":  [str, ...]      # K distinct hotpot questions
                                    # answerable from the docs in prefix
      "prefix_token_len": int,
    }

Use case: the `multi_doc_qa` scenario in
``benchmarks/bs_kernel/sglang_workloads.py`` — shared prefix = `prefix`,
K branches = K different questions over the same doc-set. Distinct
from ``build_hotpotqa_prompts.py`` (which produces one question per
prompt, for plain beam search).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"

DEFAULT_OUT = Path("data/hotpotqa_multi_q.jsonl")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_TARGET_TOKENS = 45000  # safely above L_p=40000
DEFAULT_N_PROMPTS = 8
DEFAULT_QUESTIONS_PER = 64  # so K-override up to 64 just slices


def build_prompt(
    rng: random.Random,
    examples,
    tokenizer,
    target_tokens: int,
    n_questions: int,
    cap_prefix_tokens: int | None = None,
) -> tuple[str, list[str], int]:
    """Build (prefix_text, questions_list, prefix_token_len).

    Strategy: pick ``n_questions`` distinct examples and always collect
    one question per "primary" example. Include each example's 10
    paragraphs (deduped by title) as documents, but stop adding docs
    once the running token count reaches ``cap_prefix_tokens`` (if set).
    The questions list is independent of how many docs were actually
    included — so K can exceed the doc-set size, keeping the prefix at a
    natural document boundary near the cap (used for the K=128 cells,
    where 128 full doc-sets would otherwise blow past the model's
    context window). If the prefix is still below ``target_tokens`` after
    primaries, top up with filler docs (subject to the same cap).
    """
    if n_questions > len(examples):
        raise ValueError(f"need n_questions <= {len(examples)} examples")

    indices = list(range(len(examples)))
    rng.shuffle(indices)
    primary = indices[:n_questions]
    fillers = indices[n_questions:]

    parts: list[str] = []
    seen_titles: set[str] = set()
    doc_no = 0
    questions: list[str] = []
    tl = 0  # running token count of ``parts``

    def add_example_docs(ex):
        """Add one example's deduped paragraphs, maintaining ``tl``.

        Returns early once the cap is reached so the prefix lands on a
        document boundary rather than mid-document."""
        nonlocal doc_no, tl
        ctx = ex["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            if cap_prefix_tokens is not None and tl >= cap_prefix_tokens:
                return
            if title in seen_titles:
                continue
            seen_titles.add(title)
            body = " ".join(sents).strip()
            if not body:
                continue
            doc_no += 1
            chunk = f"Document {doc_no}: {title}\n{body}\n"
            parts.append(chunk)
            tl += len(tokenizer(chunk, add_special_tokens=False).input_ids)

    # 1. Always collect every primary's question; add its docs only
    #    while under the cap.
    for idx in primary:
        ex = examples[idx]
        questions.append(ex["question"])
        add_example_docs(ex)

    # 2. Top up with filler docs until target_tokens (or cap) is reached.
    for idx in fillers:
        if tl >= target_tokens:
            break
        if cap_prefix_tokens is not None and tl >= cap_prefix_tokens:
            break
        add_example_docs(examples[idx])

    prefix = "".join(parts)
    tl = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    return prefix, questions, tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="tokenizer model id (for accurate token counts)")
    ap.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS,
                    help="number of distinct doc-sets to emit")
    ap.add_argument("--questions-per", type=int, default=DEFAULT_QUESTIONS_PER,
                    help="number of questions per doc-set (use slice for "
                         "K-override at bench time)")
    ap.add_argument("--cap-prefix-tokens", type=int, default=None,
                    help="stop adding docs once the prefix reaches this many "
                         "tokens (still collects all --questions-per "
                         "questions). Use to keep K=128 prefixes near the "
                         "model context window instead of ~175k.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/work/u4320956/hf"))
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_home
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
            prefix, questions, tl = build_prompt(
                rng, examples, tokenizer,
                args.target_tokens, args.questions_per,
                cap_prefix_tokens=args.cap_prefix_tokens,
            )
            f.write(json.dumps({
                "prefix": prefix,
                "questions": questions,
                "prefix_token_len": tl,
            }) + "\n")
            written += 1
            print(f"  [{i+1}/{args.n_prompts}] prefix_token_len={tl} "
                  f"n_questions={len(questions)}", file=sys.stderr)

    print(f"\nWrote {written} prompts to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
