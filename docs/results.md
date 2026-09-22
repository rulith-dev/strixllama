# Measured results

Everything here was measured on the target machine, against the production runtime, with the command
shown. Where a number moved for a reason other than the change being tested — usually draft
acceptance — it says so. See [measuring.md](measuring.md) before comparing any two of these.

## Machine

| | |
|---|---|
| CPU / GPU | Ryzen AI Max+ 395, Radeon 8060S (gfx1151) |
| Memory | 128 GB unified, **96 GB carved to the GPU** (Windows is left 31.6 GB) |
| Power | 140 W |
| OS | Windows 11 |
| Runtime | `pwilkin/llama.cpp` @ `f5daaa3` + this patch set, built against TheRock ROCm 10.1 |
| Model | Qwen3.8-Flash-Next 125B-A6B, Unsloth UD-IQ4_XS, 93.7 GB |
| Server flags | ctx 262144, batch/ubatch 8192, `-np 1`, flash attention on, KV f16, MTP draft, `--load-mode none --lazy-mode on-direct` |

The carve is not a detail. At 64 GB the model does not fit, ~9.8 GB spills into shared memory, and
the driver pages during decode: prefill 839 vs 942 t/s, decode 49.6 vs 38.6 ms/token, multi-turn
17-19 vs 26-30 tok/s — and that comparison was already in the 64 GB arm's favour, at 140 W against
120 W. **Set the carve before tuning anything else.**

## Headline

Measured 2026-09-19 on the shipped build, at temperature 0, `cache_prompt: false` so each run pays a
full prefill, **on a freshly started server** (see the image note below — that qualifier is load-bearing):

| | | |
|---|---|---|
| prefill, 85K tokens | **886 t/s** | 886.2 / 883.7 / 887.6 over 3 runs |
| decode, 85K context | **28.7 ms/token** (34.9 tok/s) | 29.45 / 28.25 / 28.25, draft acceptance 68-69%, 2.99 tokens per pass |
| decode, short context | **26.8 ms/token** (37.4 tok/s) | 27.47 / 26.37 / 26.42, acceptance 60%, 2.70 tokens per pass |
| image input | works | Qwen3-VL projector, 904 MB |

For reference, the same model in LM Studio on this machine decodes at about 18 tok/s.

The first run of each group is a warm-up: 29.45 against 28.25 twice, 27.47 against 26.4 twice. The
prefill figures need no such caveat — 886.2 / 883.7 / 887.6 is the tightest spread in this document,
and that is what `GGML_HIP_ENABLE_UNIFIED_MEMORY=0` bought.

### A slot that has served an image decodes ~7% slower until it is cleared

Measured on one server, same prompt, before and after running `tools/vision_stress.py`:

| | decode at 85K | acceptance |
|---|---|---|
| fresh server | 28.7 ms/token | 68-69%, 2.99 per pass |
| after an image conversation | 30.3 ms/token | 67-69%, 2.97 per pass |

Acceptance is identical, so this is a real per-pass cost and not the speculation lottery. The cause
is the image guard: M-RoPE decouples cell indices from positions, which the block-key cache's
staleness tracking cannot represent, so the cache is switched off for any memory that has seen an
image — and it stays off until a **full** clear, because an ordinary whole-sequence removal also
happens during slot reuse while the cached prefix is deliberately kept. The block-key cache is worth
about 6.5%, which is exactly what this costs.

This is the right trade (without it, images abort the server outright), but it means **a benchmark
run on a server that has served an image is not comparable to one that has not.** An earlier draft of
this file reported 32.1 ms/token at 85K for precisely that reason.

The command, against the running production server — an 85K-token prefix of real prose followed by a
question, which is what makes the draft-acceptance figure mean anything (see below):

```bash
python tools/decode_lab.py --keep --words 700,28800 --n 64
```

`--keep` probes whatever server is already listening rather than launching one, so this measures
production itself. The depth probe uses random words; for the acceptance-sensitive numbers above,
point `--gen-prefix` at a long document of your own and use `--gen`:

```bash
python tools/decode_lab.py --config base --gen 400 --gen-prefix <a long document> --gen-prefix-chars 85000
```

**`--gen-prefix-chars` is characters, not tokens**, and the ratio is entirely
language-dependent — the corpus used here is Chinese and runs at almost exactly 1 token per
character, where English prose is nearer 1 per 4. Ask the server (`POST /tokenize`) instead of
assuming; a prefix meant to be 85K tokens was briefly 340K, which the server rejected outright.

## How decode got from 21.5 to ~31 tok/s at 85K

In order, each measured when it landed. These are not additive — later entries were measured on top
of earlier ones, and acceptance drifts between them.

| change | effect |
|---|---|
| QSA block-key cache | 97K decode 63.5 → 58.5 ms/token; context slope 0.241 → 0.182 ms per 1000 tokens |
| sparse attention at decode (gather) | 97K decode 58.9 → 47.9 ms/token; slope 0.188 → 0.078 |
| draft head with its own IQ4_XS output projection | +5% English, +9% Chinese; acceptance unchanged |
| HIP graphs keyed on shape | +2%, and every draft graph replays instead of resetting its neighbour |
| `draft_max` 5 → 3 | 30.2 → 27.9 ms/token (see [decode-budget.md](decode-budget.md)) |

The two attention changes are the substantial ones, and they are the same shape of bug: work that
only prefill's layout supported, so a single decoded token fell back to reading the entire cache.

## Prefill

| | |
|---|---|
| IQ3_S reaching the matrix cores | 830.6 → **876.7 t/s** (+5.6%), quality unchanged |
| `GGML_HIP_ENABLE_UNIFIED_MEMORY=0` | 867.1 → 903.6 t/s (+4.2%), and run-to-run spread collapses 6× |
| overlapped PLE gather (depth 1 → 16) | gather 3.08 → 2.28 s, but end-to-end a wash: a detached prefetch thread was already hiding it |

The unified-memory switch was the largest prefill win of its day and is not a code change: with it
on, allocations spill into shared GPU memory while the carve still has room, and shared memory is
the same system RAM the PLE page cache needs.

## Correctness

Two bugs that produced wrong output rather than slow output, both found late because the standard
gate could not see them:

- **Speculative verification ran dense attention with no causal mask.** Only the sparse kernel
  honours "indices, no mask", and it declines batches under 128 queries; on this backend every
  smaller batch fell through to the generic kernels. Single-token decode had nothing to leak, but a
  2-8 token verification batch let the target see the very tokens it was checking, so acceptance
  stopped being a check: long answers drifted, looped, and stopped early. Perplexity at ctx 2048
  read 2.3807 in every variant, because the bug only bites once sparse attention is active above
  ~2K. At ctx 4096 it is unmissable: 1.0841 at ubatch 512 and 1.8397 at ubatch 4, against 2.4646
  fixed. After the fix, 1500 speculative tokens are byte-identical to greedy decoding.
- **Image input aborted the server three separate ways**, all in the QSA block machinery, all in
  `llama-memory-hybrid-idx.cpp`. M-RoPE repeats one position across an image and advances positions
  past it by the image's extent, so cells and positions decouple in both directions, and the
  block-key cache's staleness tracking is keyed on 1-D positions. The third abort was in the *MTP
  draft* context, which never stores image tokens and so develops a hole its positions run past.

  None of the three reproduced on a single-turn test at short context. Two needed the third or
  fourth turn of a growing image conversation and one needed a divergent branch, so
  `tools/vision_stress.py` walks a conversation to ~17K tokens and then branches twice. Run it
  against any change near sparse attention, M-RoPE or the KV cache; it generates its own image, so
  it needs nothing but a server with a projector loaded.

## What is still open

- Graph reuse on speculative decodes is ~45%: the draft context alternates between two batch shapes
  against one cached graph result. More than one live result needs scheduler surgery.
- `set_input_kq_mask` scans all 85K cells once per decode (~0.9 ms).
- About 43 ms of the 85K pass is weight bandwidth and does not move without changing the file.
- Decode has roughly 2.5× of unused machine *with speculation off*: 1 stream 19.0 tok/s aggregate,
  2 streams 30.6, 4 streams 47.1, because the weight bytes are read once per batch regardless of
  how many sequences share it. It is the same lever speculation pulls for a single user, and the
  two do not stack: **with MTP on, four streams give 57.7 tok/s against 37.5 for one (1.54×)**,
  each stream at 14 tok/s, and eight streams add nothing (62.9). A step costs ~30 ms fixed plus
  ~1 ms per distinct routed expert it touches (512 experts, 10 per token, 116 MB each), so
  draft tokens and other users' tokens draw on the same budget. Before `apply_iq3s_mmq_hip` that
  made MTP lose past four users (62.9 against 71.4 at eight streams); with it the two tie (71.8
  against 72.7), and a per-step draft budget that thins the draft under load measured nothing, so
  the fixed draft of 3 stays at every stream count (`docs/results/concurrency-mtp-20260921.json`). Measured per
  kernel afterwards, the expert path was already near the ceiling in the one-user configuration;
  what was slow was IQ3_S through MMQ (the multi-user batch sizes) and a tiling cliff above 16
  tokens per step — `apply_iq3s_mmq_hip` fixes both: eight users with MTP 60.5 → 70.1 tok/s.
- **Several users at depth.** A ubatch that serves several slots carries several sequences; until
  `apply_multi_stream_qsa` the sparse path declined it and ran dense attention over the whole used
  pool, so four users each 35K deep got 28.3 tok/s together at 99 ms/token, less than one user alone
  (31.7). With mixed ubatches on the sparse path they get 33–39 tok/s at 70–87 ms/token, two users
  34 (was 22), and one user is 7% faster at depth (26.8 ms/token) because a multi-stream step no
  longer switches the block-key cache off for good. The mixed reserve graph lost its dense mask with
  it: four slots at 262144 × 8192 load and cost 1.3 GB over one slot (4096 used to cost 9 GB and
  8192 did not load), so the default `-np 1` is only about the ~1 GB and the slower prefill of a
  shared pool. Image input needs a single slot.
- **What the prompt cache cost in system RAM.** `--cache-ram` defaults to 8192 MiB and the manager never
  set it, so the server held ~8 GB of system memory for cached conversation state on a machine whose GPU
  carve leaves 31.6 GB for the whole desktop — the KV cache is in the carve, this was the RAM tier above
  the disk one. Turning it down needed a change first: `alloc()` refuses a state larger than the limit and
  `persist()` only ever ran on a state `alloc()` had accepted, so a small limit would have quietly stopped
  every long conversation from reaching disk. With `persist()` taking the buffers directly and
  `--cache-ram 1024`, the same four conversations cycled through one slot leave the server at
  **4.13 GB resident instead of 8.07 GB**, and finish 83.4 s → 57.5 s faster because the disk reads
  stop competing with the cache for memory. Two defects surfaced with it: the disk tier
  ranked candidates by `f_keep`, so a short entry that was a complete prefix (`f_keep = 1.000`) shadowed
  every longer entry of the same conversation — the 79K-token chat loaded the 49370-token entry and
  prefilled 30071 tokens (46.3 s) where the right entry costs 2897 (**17.3 s, against 96.6 s cold**) — and
  reopening any cached conversation with MTP **off** hit `GGML_ASSERT(ctx_dft)` and killed the server. An
  entry written with MTP on is now usable without it (the draft half is dropped); the reverse is refused,
  because nothing can prime a draft context for a sequence the target is already deep into. The disk
  ceiling is a setting now, since one long conversation is 5.6 GB. Details:
  `docs/results/prompt-cache-20260922.json`.
- **Why a cache hit still cost 17 s.** Of a hit on the 79K-token conversation, 12.1 s was reading the
  5.630 GiB entry and 5.5 s was replaying 2897 tokens. The read ran at 505 MB/s; the same file read
  with `FILE_FLAG_NO_BUFFERING` comes off the drive at **3451 MB/s**, and a copy written in one pass
  reads no faster, so neither fragmentation nor the drive explains it. It was the cached I/O path:
  every GiB is copied into the page cache on the way past, which buys nothing for data that is read
  once and handed to the GPU, and costs dearly when the carve leaves 31.6 GB of system memory. Reading
  through one aligned staging buffer gives **1656 ms at 3482 MB/s** and the hit drops to **8.1 s**
  against 96.6 s cold. The replay is now the larger half and is inherent: sampling needs logits for the
  last token, and a GDN recurrent state cannot be rewound by one token, so the server restores the
  nearest context checkpoint and replays from there. Those checkpoints are also ~3.3 GB of the 4.15 GB
  the server holds with one long conversation resident. Details:
  `docs/results/prompt-cache-20260922.json`.
