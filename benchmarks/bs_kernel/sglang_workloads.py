"""Build TreeSpec workloads from real text matching the four SGLang
multi-* benchmark scenarios.

The four scenarios are mapped onto the 2-level (shared + per-leaf
private) ``TreeSpec`` contract from ``tree_driver.py``:

  scenario              | B   | K  | shared prefix      | per-leaf private prefix
  ----------------------|-----|----|--------------------|------------------------
  multi_level_system    | 4   | 32 | sys (~4 K tok)     | GSM8K question (80 tok)
  multi_few_shot        | 8   | 16 | sys + fewshot      | GSM8K question (80 tok)
  multi_chain_reasoning | 32  | 4  | sys + 1 question   | (empty — fork at decode)
  multi_document        | 16  | 8  | 4 docs (~4.4 K)    | GSM8K question (80 tok)

Total leaves: 128 per scenario (parity with the kernel-level test).
Question/sys/fewshot/doc token lengths are *padded or truncated to a
fixed length per role* so each PromptGroup's K leaves share a uniform
``current_pos`` — the existing backend.plan_decode_step contract
requires this. Padding uses each leaf's last real token (kernel-fair;
content doesn't affect timing).

Data inputs:
  * system_prompt.template under
    ``3rdparty/FastTree-Artifact/sglang_v0.2.13_bench/benchmarks/data/``
    (committed). Rendered with (LOCATION, LANGUAGE) variants.
  * GSM8K ``test.jsonl`` — downloaded on first call if missing.

Multi-document doesn't load the actual Llama-2 paper (which would
need PyPDF2 + a 13 MB download). Instead we synthesize each doc-bundle
from rolled-up GSM8K Q&A text, matching the per-leaf token-length
profile from the kernel test (~4400 tok shared, ~80 tok private).
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from beam_engine.tree_driver import PromptGroup, TreeSpec


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SGLANG_DATA = (
    _REPO_ROOT / "3rdparty" / "FastTree-Artifact"
    / "sglang_v0.2.13_bench" / "benchmarks" / "data"
)
_SYSTEM_PROMPT_TEMPLATE = _SGLANG_DATA / "system_prompt.template"
_GSM8K_PATH = _SGLANG_DATA / "test.jsonl"
_GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math"
    "/master/grade_school_math/data/test.jsonl"
)


# Target token lengths.
#
# Historically these were padded to fixed sizes to match the kernel-level
# test in ``bench_sglang_tree_shapes.py``. To replicate the FastTree
# paper's "exact" setup faithfully (Hu et al., MLSys'25, §4.1), the
# ``paper_exact`` variants below use the rendered template / 20-shot
# example bundles / Llama-3-paper segments at their *natural* length.
# The fixed-length constants are kept for the legacy / parity path.
SYS_LEN = 4096
QUESTION_LEN = 80
FEWSHOT_LEN = 2560
DOC_BUNDLE_LEN = 4400
CHAIN_PROMPT_PRIVATE_LEN = 80   # the question part of "sys + question"

# FastTree paper §4.1 — multi-chain CoT prefix library
# (verbatim from ``3rdparty/FastTree-Artifact/.../bench_multi_chain_reasoning.py:prompt_lib``).
CHAIN_PROMPT_LIB: tuple[str, ...] = (
    "Let us think step by step.",
    "Approach this methodically. Let's dissect the problem into smaller, more manageable parts.",
    "It's important to proceed step by step, ensuring accuracy at each stage.",
    "Take a deep breath and break this down.",
    "A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem.",
    "I am extremely good at math.",
)
# 20-shot fewshot per the FastTree paper.
FEWSHOT_NUM_SHOTS = 20
# Doc-bundle config: 30 segments of ~1100 tok from the Llama-3 paper,
# 4 docs sampled per branch, 16 branches × 8 questions = 128 leaves.
DOC_SEGMENTS_PATH = _REPO_ROOT / "benchmarks" / "bs_kernel" / "data" / "llama3_segments.json"
DOC_NUM_DOCS_PER_BRANCH = 4
# Hand-picked Llama-3-paper questions for multi_document (analogue of the
# 10 Llama-2 questions hard-coded in the FastTree build_doc_dataset.py).
DOC_QUESTIONS = (
    "What are the parameter counts of the Llama 3 model family?",
    "How many tokens of pre-training data does Llama 3 use?",
    "What context length does the Llama 3 model family support?",
    "Which group-query attention configuration does Llama 3 use?",
    "What is the vocabulary size of the Llama 3 tokenizer?",
    "What is the SFT (supervised fine-tuning) dataset size?",
    "Which RLHF method does Llama 3 use?",
    "What is the activation function used in Llama 3?",
    "What scaling law experiments motivated the model sizes?",
    "Which safety benchmarks does Llama 3 evaluate against?",
)


def _ensure_gsm8k() -> None:
    if _GSM8K_PATH.exists():
        return
    _GSM8K_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_GSM8K_URL, _GSM8K_PATH)


def _render_sys(location: str, language: str) -> str:
    text = _SYSTEM_PROMPT_TEMPLATE.read_text()
    return text.format(LOCATION=location, LANGUAGE=language)


def _gsm8k_questions(tokenizer, n_min: int = 256) -> list[list[int]]:
    """Return at least ``n_min`` GSM8K questions as token-id lists,
    each truncated/padded to QUESTION_LEN tokens."""
    _ensure_gsm8k()
    out: list[list[int]] = []
    with _GSM8K_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tokenizer.encode(row["question"], add_special_tokens=False)
            out.append(_pad_or_truncate(ids, QUESTION_LEN))
            if len(out) >= n_min:
                break
    if len(out) < n_min:
        # Repeat to reach n_min if GSM8K runs short (unlikely).
        out = (out * ((n_min + len(out) - 1) // len(out)))[:n_min]
    return out


def _gsm8k_qa_concatenated(tokenizer, target_len: int) -> list[int]:
    """Concatenate GSM8K Q+A pairs until we hit ``target_len`` tokens,
    then truncate (used to synthesize fewshot bundles and doc bundles).
    """
    _ensure_gsm8k()
    ids: list[int] = []
    with _GSM8K_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) >= target_len:
                break
    return _pad_or_truncate(ids, target_len)


def _pad_or_truncate(ids: list[int], target: int) -> list[int]:
    if len(ids) >= target:
        return ids[:target]
    if not ids:
        return [0] * target
    return ids + [ids[-1]] * (target - len(ids))


def _sys_tokens(tokenizer, location: str, language: str) -> list[int]:
    text = _render_sys(location, language)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return _pad_or_truncate(ids, SYS_LEN)


def _sys_tokens_raw(tokenizer, location: str, language: str) -> list[int]:
    """Render the Meta AI system prompt template with the given variant
    and tokenize — *without* padding to a fixed length. Used in the
    paper-exact path so the workload reflects the real template length
    (Llama-3 tokenizer → ~2 516 tok), matching FastTree §4.1's table
    structure (459 + 38 + 584 + 2 112 tok with Llama-2 tokenizer)."""
    return tokenizer.encode(_render_sys(location, language),
                            add_special_tokens=False)


def _gsm8k_raw_questions(n_min: int) -> list[dict[str, str]]:
    """Return at least ``n_min`` raw GSM8K rows (untokenised). The
    paper-exact ``multi_few_shot`` and ``multi_chain_reasoning`` builders
    need the full Q+A text, not just tokenised questions."""
    _ensure_gsm8k()
    out: list[dict[str, str]] = []
    with _GSM8K_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if len(out) >= n_min:
                break
    if len(out) < n_min:
        out = (out * ((n_min + len(out) - 1) // len(out)))[:n_min]
    return out


# ---------------------------------------------------------------------------
# Per-scenario builders — return TreeSpec
# ---------------------------------------------------------------------------


_SYS_VARIANTS = [
    ("the United States", "English"),
    ("the United States", "Spanish"),
    ("Canada", "English"),
    ("China", "Chinese"),
]


def build_multi_level_system(
    tokenizer,
    k_override: int | None = None,
    b_override: int | None = None,
    paper_exact: bool = False,
) -> TreeSpec:
    """B sys-prompts × K questions each. Natural (B=4, K=32) = 128
    leaves. ``k_override`` flips kernel phase; ``b_override`` lets B
    exceed the 4 base ``_SYS_VARIANTS`` (we cycle through them — content
    is kernel-fair, the (LOC, LANG) pairs just keep neighbouring groups
    text-distinct).

    ``paper_exact=True`` uses the rendered Meta-AI sys-prompt template
    at its natural token length (no pad to SYS_LEN). This matches the
    FastTree paper §4.1 (A) which uses the same template with random
    LOCATION/LANGUAGE replacements; the paper reports 459+38+584+2112
    = 3193 tok with Llama-2 tokenizer; Llama-3 tokenizer yields
    ~2 516 tok for the same template.
    """
    K = k_override if k_override is not None else 32
    B = b_override if b_override is not None else len(_SYS_VARIANTS)
    questions = _gsm8k_questions(tokenizer, n_min=B * K)
    sys_fn = _sys_tokens_raw if paper_exact else _sys_tokens
    groups: list[PromptGroup] = []
    for b in range(B):
        loc, lang = _SYS_VARIANTS[b % len(_SYS_VARIANTS)]
        shared = sys_fn(tokenizer, loc, lang)
        priv = [questions[b * K + k] for k in range(K)]
        groups.append(PromptGroup(shared_prefix_ids=shared,
                                  private_prefix_ids_per_leaf=priv))
    return TreeSpec(groups=groups)


def build_multi_few_shot(
    tokenizer,
    k_override: int | None = None,
    b_override: int | None = None,
    paper_exact: bool = False,
) -> TreeSpec:
    """1 sys × 8 fewshot bundles × 16 questions = 128 leaves.

    Encoded as B=8, K=16: each PromptGroup's shared = sys + fewshot[i].
    The cross-group sys is duplicated 8x; matches the kernel test's
    bs_kernel pick (SHARED_2L_DEC_TAIL, depth=2) and gives the picker
    the same workload shape it saw at the kernel level. ``k_override``
    flips the beam count for paper sweeps (e.g. K=32 to push the
    prompt-root nodes into FastTree's phase 0).

    ``paper_exact=True`` replicates FastTree §4.1 (B): exactly 20-shot
    examples per bundle, non-overlapping stride 36 over GSM8K
    (matching ``bench_multi_few_shot.py``). Each bundle's text length
    is natural (no pad to FEWSHOT_LEN); private prefix is the raw
    "Question: …\\nAnswer:" prompt (not just the question tokens).
    """
    B = b_override if b_override is not None else 8
    K = k_override if k_override is not None else 16
    sys_fn = _sys_tokens_raw if paper_exact else _sys_tokens
    sys = sys_fn(tokenizer, *_SYS_VARIANTS[0])
    if paper_exact:
        # Stride = num_shots + num_questions per bench_multi_few_shot.py.
        stride = FEWSHOT_NUM_SHOTS + K
        rows = _gsm8k_raw_questions(B * stride)
        groups: list[PromptGroup] = []
        for b in range(B):
            offset = stride * b
            # 20-shot examples: "Question: …\nAnswer: …\n\n" × 20
            fewshot_text = ""
            for i in range(FEWSHOT_NUM_SHOTS):
                fewshot_text += (
                    f"Question: {rows[offset + i]['question']}\n"
                    f"Answer: {rows[offset + i]['answer']}\n\n"
                )
            fewshot_ids = tokenizer.encode(fewshot_text, add_special_tokens=False)
            shared = sys + fewshot_ids
            # Private prefix per leaf: "Question: <q>\nAnswer:"
            # tree_batch_decode requires uniform-length privates within a
            # group — pad to the max length seen by repeating the last
            # token (kernel-fair; matches _pad_or_truncate semantics).
            priv_raw: list[list[int]] = []
            for j in range(K):
                q = rows[offset + FEWSHOT_NUM_SHOTS + j]["question"]
                priv_raw.append(tokenizer.encode(
                    f"Question: {q}\nAnswer:", add_special_tokens=False,
                ))
            max_len = max(len(p) for p in priv_raw)
            priv = [_pad_or_truncate(p, max_len) for p in priv_raw]
            groups.append(PromptGroup(
                shared_prefix_ids=shared, private_prefix_ids_per_leaf=priv,
            ))
        return TreeSpec(groups=groups)

    questions = _gsm8k_questions(tokenizer, n_min=B * K)
    # Make 8 distinct fewshot bundles (use different offsets into GSM8K).
    # Each is FEWSHOT_LEN tokens.
    fewshots: list[list[int]] = []
    with _GSM8K_PATH.open() as f:
        all_lines = [l for l in f if l.strip()]
    rng = random.Random(0)
    for i in range(B):
        rng.shuffle(all_lines)
        text_acc = ""
        for line in all_lines:
            row = json.loads(line)
            text_acc += f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
            if len(tokenizer.encode(text_acc, add_special_tokens=False)) >= FEWSHOT_LEN:
                break
        ids = tokenizer.encode(text_acc, add_special_tokens=False)
        fewshots.append(_pad_or_truncate(ids, FEWSHOT_LEN))
    groups: list[PromptGroup] = []
    for b in range(B):
        shared = sys + fewshots[b]
        priv = [questions[b * K + k] for k in range(K)]
        groups.append(PromptGroup(shared_prefix_ids=shared,
                                  private_prefix_ids_per_leaf=priv))
    return TreeSpec(groups=groups)


def build_multi_chain_reasoning(
    tokenizer,
    k_override: int | None = None,
    b_override: int | None = None,
    paper_exact: bool = False,
) -> TreeSpec:
    """B (sys+question) × K chains. Natural (B=32, K=4) = 128 leaves.

    Default mode: K leaves under each group share an identical prefix
    (sys+question) with *empty* per-leaf private — chains diverge via
    top-K at prefill (fork_at_prefill mode in tree_batch_decode).

    ``paper_exact=True`` replicates FastTree §4.1 (C): each chain has
    its own non-empty private prefix of the form "Answer: <prompt_lib[i
    % 6]>" so the chains diverge at the *prefill private* level rather
    than via fork-at-prefill argmax. This matches
    ``bench_multi_chain_reasoning.py`` where ``s.fork(K)`` extends each
    chain with one of 6 CoT prefixes before generation.

    ``k_override`` is the chain count; at K=32 the prompt-root nodes
    cross into FastTree's phase 0. ``b_override`` sets the number of
    distinct GSM8K questions; each question is forked into K chains.
    """
    B = b_override if b_override is not None else 32
    K = k_override if k_override is not None else 4
    sys_fn = _sys_tokens_raw if paper_exact else _sys_tokens
    sys = sys_fn(tokenizer, *_SYS_VARIANTS[0])
    if paper_exact:
        rows = _gsm8k_raw_questions(B)
        chain_prefix_ids = [
            tokenizer.encode(f"Answer: {p}", add_special_tokens=False)
            for p in CHAIN_PROMPT_LIB
        ]
        # Pad to uniform length (TreeSpec.validate requires uniform per-leaf).
        chain_max = max(len(p) for p in chain_prefix_ids)
        chain_prefix_ids = [_pad_or_truncate(p, chain_max) for p in chain_prefix_ids]
        groups: list[PromptGroup] = []
        for b in range(B):
            q_text = f"Question: {rows[b]['question']}\n"
            shared = sys + tokenizer.encode(q_text, add_special_tokens=False)
            priv = [chain_prefix_ids[i % len(CHAIN_PROMPT_LIB)] for i in range(K)]
            groups.append(PromptGroup(
                shared_prefix_ids=shared, private_prefix_ids_per_leaf=priv,
            ))
        return TreeSpec(groups=groups)

    questions = _gsm8k_questions(tokenizer, n_min=B)
    groups = []
    for b in range(B):
        shared = sys + questions[b]
        priv = [[] for _ in range(K)]  # empty → fork_at_prefill mode
        groups.append(PromptGroup(shared_prefix_ids=shared,
                                  private_prefix_ids_per_leaf=priv))
    return TreeSpec(groups=groups)


def build_multi_document(
    tokenizer,
    k_override: int | None = None,
    b_override: int | None = None,
    paper_exact: bool = False,
) -> TreeSpec:
    """B doc-bundles × K questions. Natural (B=16, K=8) = 128 leaves.

    Each doc-bundle is DOC_BUNDLE_LEN tokens, sampled from GSM8K Q&A
    text (real text, kernel-equivalent to the Llama-2 paper segments
    the SGLang bench uses). ``k_override`` flips the per-doc beam
    count; ``b_override`` controls how many doc bundles are generated.
    """
    B = b_override if b_override is not None else 16
    K = k_override if k_override is not None else 8
    if paper_exact:
        # FastTree §4.1 (D) replicas: 4 random docs/branch from a pool of
        # ~30 segments cut from the Llama-3 paper (no sys prompt), 8
        # hand-picked questions per branch.
        if not DOC_SEGMENTS_PATH.exists():
            raise FileNotFoundError(
                f"paper_exact multi_document needs Llama-3 paper segments at "
                f"{DOC_SEGMENTS_PATH}. Run "
                f"`benchmarks/bs_kernel/build_llama3_segments.py` first."
            )
        payload = json.loads(DOC_SEGMENTS_PATH.read_text())
        segments: list[str] = payload["documents"]
        rng = random.Random(0)
        # Pre-tokenise each segment once.
        seg_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in segments
        ]
        # Branch preamble per bench_multi_document.py.
        preamble_ids = tokenizer.encode(
            "Please answer a question according to given documents.\nDocuments begin.\n",
            add_special_tokens=False,
        )
        end_ids = tokenizer.encode("\nDocuments end.", add_special_tokens=False)
        groups: list[PromptGroup] = []
        for b in range(B):
            doc_idx = sorted(rng.sample(range(len(seg_ids)), DOC_NUM_DOCS_PER_BRANCH))
            shared: list[int] = list(preamble_ids)
            for di in doc_idx:
                shared.extend(seg_ids[di])
            shared.extend(end_ids)
            priv_raw: list[list[int]] = []
            for j in range(K):
                q = DOC_QUESTIONS[j % len(DOC_QUESTIONS)]
                priv_raw.append(tokenizer.encode(
                    f"\n\nBased on the above documents, please answer this question:\n"
                    f"Q{j}: {q}\nAnswer in three words or fewer.",
                    add_special_tokens=False,
                ))
            max_len = max(len(p) for p in priv_raw)
            priv = [_pad_or_truncate(p, max_len) for p in priv_raw]
            groups.append(PromptGroup(
                shared_prefix_ids=shared, private_prefix_ids_per_leaf=priv,
            ))
        return TreeSpec(groups=groups)

    questions = _gsm8k_questions(tokenizer, n_min=B * K)
    _ensure_gsm8k()
    with _GSM8K_PATH.open() as f:
        all_lines = [l for l in f if l.strip()]
    rng = random.Random(1)
    groups = []
    for b in range(B):
        rng.shuffle(all_lines)
        text_acc = ""
        for line in all_lines:
            row = json.loads(line)
            text_acc += f"Doc: {row['question']}\n{row['answer']}\n\n"
            if len(tokenizer.encode(text_acc, add_special_tokens=False)) >= DOC_BUNDLE_LEN:
                break
        ids = tokenizer.encode(text_acc, add_special_tokens=False)
        shared = _pad_or_truncate(ids, DOC_BUNDLE_LEN)
        priv = [questions[b * K + k] for k in range(K)]
        groups.append(PromptGroup(shared_prefix_ids=shared,
                                  private_prefix_ids_per_leaf=priv))
    return TreeSpec(groups=groups)


def build_multi_chain_reasoning_stage2(
    tokenizer, *, stage1_chain_len: int = 256, b_override: int | None = None,
    num_chains: int = 4,
) -> TreeSpec:
    """Stage-2 of the two-stage FastTree §4.1 (C) multi_chain_reasoning
    workload — the post-fork "majority vote" decode.

    Each of the B questions runs a single (K=1) decode that attends to:
      sys + "Question: <q>\\n" + majority-intro + K × ("Solution N: " +
      ``stage1_chain_len`` placeholder tokens) + final-answer prompt.

    The K = ``len(CHAIN_PROMPT_LIB)`` doesn't matter here — by stage 2,
    the chain outputs have been concatenated into the prompt as text.
    We use the natural K from FastTree's bench (4 chains) so the
    placeholder count = 4 * stage1_chain_len.

    Placeholder tokens are kernel-fair: stage 2's attention timing
    depends on KV length + count of attending query rows, not on the
    semantic content of the placeholders.
    """
    B = b_override if b_override is not None else 32
    sys = _sys_tokens_raw(tokenizer, *_SYS_VARIANTS[0])
    rows = _gsm8k_raw_questions(B)

    majority_intro = tokenizer.encode(
        "Answer: To answer this question, here are some possible solutions. "
        "After considering all of them, I will do a majority vote.\n\n",
        add_special_tokens=False,
    )
    solution_header_ids = [
        tokenizer.encode(f"Solution {i+1}: ", add_special_tokens=False)
        for i in range(num_chains)
    ]
    solution_footer = tokenizer.encode("\n\n", add_special_tokens=False)
    final_prompt = tokenizer.encode(
        "\nBy considering the above solutions and doing a majority vote, "
        "I think the final answer (a single integer number) is ",
        add_special_tokens=False,
    )
    # Placeholder chain output: stage1_chain_len tokens. Use 0 (pad-token
    # equivalent); kernel-fair.
    chain_placeholder = [0] * stage1_chain_len

    groups: list[PromptGroup] = []
    for b in range(B):
        q_ids = tokenizer.encode(
            f"Question: {rows[b]['question']}\n", add_special_tokens=False,
        )
        shared: list[int] = list(sys) + q_ids + majority_intro
        for sh in solution_header_ids:
            shared.extend(sh)
            shared.extend(chain_placeholder)
            shared.extend(solution_footer)
        # Put the final-prompt as the (K=1) leaf's private prefix so
        # tree_batch_decode goes through the standard (non-fork) path —
        # fork_at_prefill with K=1 + non-page-aligned L_s causes
        # fasttree's plan_decode_step to read an empty pages_tail.
        groups.append(PromptGroup(
            shared_prefix_ids=shared,
            private_prefix_ids_per_leaf=[list(final_prompt)],
        ))
    return TreeSpec(groups=groups)


SCENARIO_BUILDERS = {
    "multi_level_system":    build_multi_level_system,
    "multi_few_shot":        build_multi_few_shot,
    "multi_chain_reasoning": build_multi_chain_reasoning,
    "multi_document":        build_multi_document,
}
