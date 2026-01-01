# Cascade Inference Layout

## Input Text Structure

The input consists of variations of a sentence with hierarchical structure:

0. `<s>` As you come into this world, something else is also born -- a connection to the unseen **forces**
1. `<s>` As you come into this world, something else is also born -- a connection to the people who shape your **journey**
2. `<s>` As you come into this world, something else is also born -- a connection to the **infinite possibilities**
3. `<s>` As you come into this world, something else is also born -- an echo of your **existence**
4. `<s>` As you come into this world, something else is also born -- an echo of your **essence**
5. `<s>` As you come into this world, something else is also born. You begin your life, it begins a **journey**
6. `<s>` As you come into this world, something else is also born. You begin your life, and in that **moment**
7. `<s>` As you come into this world, something else is also born. You begin your life, and the world around **you**

## Hierarchical Tree Structure

```
<s> As you come into this world, something else is also born        Level 0
[0, 1, 2, 3] +9

├── -- a connection to the           Level 1
│   [4, 5] +4
│   ├── unseen forces [10] +1                                       Level 2
│   ├── people who will shape your journey [11, 12] +1
│   └── infinite possibilities [] +2
│
├── -- an echo of your               Level 1
│   [6, 7] +2
│   ├── existence [] +1                                             Level 2
│   └── essence [] +1
│
└── . You begin your life,           Level 1
    [8, 9] +3
    ├── it begins a journey [13] +1                                 Level 2
    ├── and in that moment [14] +1
    └── and the world around you [15] +1
```

**[page_indices] +APPEND_TOKENS**

## Query/Output Layout (Ragged Tensor)

### Level 0 View
```
| forces journey infinite possibilities existence essence journey moment you |
```

### Level 1 View
```
| forces journey infinite possibilities | existence  essence | journey moment you |
```

### Level 2 View
```
| forces | journey | infinite | possibilities | existence | essence | journey | moment | you |
```

## Paged KV-Cache Layout (page_size=4)

| Page 0 | Page 1 | Page 2 | Page 3 | Page 4 | Page 5 | Page 6 | Page 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|
| `<s>` As you come | into this world, | something else is also | born | -- a connection to | the | -- an echo of | your |
| **Page 8** | **Page 9** | **Page 10** | **Page 11** | **Page 12** | **Page 13** | **Page 14** | **Page 15** |
| . You begin your | life, | unseen | people who will shape | your | it begins a | and in that | and the world around |

## Index Mappings

### qo_indptr (Query/Output indices per level)
- **level 0:** [0, 9]
- **level 1:** [0, 4, 6, 9]
- **level 2:** [0, 1, 2, 4, 5, 6, 7, 8, 9]

### kv_page_indptr (KV page indices per level)
- **level 0:** [0, 3]
- **level 1:** [0, 2, 4, 6]
- **level 2:** [0, 1, 3, 3, 3, 3, 4, 5, 6]

### kv_page_indices (Actual page numbers)
- **level 0:** [0, 1, 2, 3]
- **level 1:** [4, 5, 6, 7, 8, 9]
- **level 2:** [10, 11, 12, 13, 14, 15]

### kv_last_page_len (Length of last page per sequence)
- **level 0:** [1]
- **level 1:** [1, 1, 1]
- **level 2:** [1, 1, 0, 0, 0, 3, 3, 4]

## Explanation

Cascade inference is a technique for efficient parallel decoding in large language models. The key idea is to organize multiple decode sequences in a hierarchical tree structure where sequences share common prefixes.

### Key Concepts:

1. **Hierarchical Structure**: Sequences are organized in levels (0, 1, 2) where each level represents different branching points in the generation tree.

2. **Shared Prefixes**: Level 0 contains the common prefix shared by all sequences. Level 1 contains the first branching points, and Level 2 contains the leaf sequences.

3. **Paged KV-Cache**: The key-value cache is organized into fixed-size pages (page_size=4 tokens), allowing efficient memory management and sharing of common prefixes.

4. **Ragged Tensor Layout**: The query/output tensors have variable lengths per level, represented as ragged tensors for efficient computation.

5. **Index Pointers**: Various index arrays (`qo_indptr`, `kv_page_indptr`, etc.) maintain the mapping between logical sequence positions and physical memory locations.

This approach enables efficient batch processing of multiple generation candidates while maximizing cache reuse for shared prefixes.