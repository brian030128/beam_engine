"""Build the document corpus for the FastTree-§4.1-(D) multi_document
scenario from the public Llama-3 docs page
(`https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/`),
saved verbatim to ``benchmarks/bs_kernel/data/llama3_docs.txt``.

The FastTree paper's recipe segments the Llama-2 paper PDF into 30 chunks
of ~1100 Llama-2 tokens each; with only the model-card docs page as a
source we get ~840 Llama-3 tokens total, so we segment at ~140 tokens
to produce ~6 distinct chunks. The multi_document builder then samples
N docs per branch from those 6 (smaller pool than the paper's 30, less
inter-branch variety, but all content is real Meta-authored Llama-3
documentation, not synthetic).

Output schema (matches FastTree's ``questions.jsonl`` layout):
    {"documents": [<6 strings>], "questions": [<10 strings>]}
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent / "data"
SRC_PATH = DATA_DIR / "llama3_docs.txt"
OUT_PATH = DATA_DIR / "llama3_segments.json"

SEGMENT_TOKEN_TARGET = 140    # per-segment target; corpus is ~840 tok → ~6 segments
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

# 10 hand-picked questions about the Llama-3 page content. Mirrors the
# FastTree build_doc_dataset.py recipe (10 hand-picked Llama-2 questions).
QUESTIONS = [
    "What token marks the beginning of a Llama 3 prompt?",
    "What token signifies the end of a message in a turn?",
    "What roles can appear between Llama 3 header tokens?",
    "What does the assistant header at the end of a prompt do?",
    "Which token is equivalent to the EOS token?",
    "Are newline characters part of the Llama 3 prompt format?",
    "How does a system message appear in a Llama 3 Instruct prompt?",
    "Does a Llama 3 prompt always end with the user message?",
    "What special token wraps the role identifier?",
    "Where can the code to generate the Llama 3 Instruct prompt be found?",
]


def main() -> int:
    text = SRC_PATH.read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"[build_llama3_segments] corpus tokens: "
          f"{len(tok.encode(text))}", flush=True)

    SEP = "\n\n"
    parts = [p for p in text.split(SEP) if p.strip()]
    print(f"[build_llama3_segments] paragraphs: {len(parts)}", flush=True)

    segments: list[str] = []
    tmp: list[str] = []
    tmp_len = 0
    for p in parts:
        tmp.append(p)
        tmp_len += len(tok.encode(p))
        if tmp_len >= SEGMENT_TOKEN_TARGET:
            segments.append(SEP.join(tmp))
            tmp = []
            tmp_len = 0
    if tmp:
        segments.append(SEP.join(tmp))

    for i, s in enumerate(segments):
        print(f"  segment {i:2d}: {len(tok.encode(s)):4d} tokens",
              flush=True)

    OUT_PATH.write_text(json.dumps(
        {"documents": segments, "questions": QUESTIONS}, indent=2,
    ))
    print(f"[build_llama3_segments] wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
