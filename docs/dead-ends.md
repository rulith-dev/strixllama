# Measured and rejected

Ideas that were built and measured, and did not earn their place. They are here so they are not
rebuilt — several look obviously right and are not.

## Speculation

**Batching the per-query gather in the sparse attention path.** Removes ~900 nodes from a 4-token
graph. Exactly neutral: 83.8 ± 6.0 ms per pass per-query against 83.6 ± 5.7 batched, same binary,
same draft, same prompt. With HIP graphs replaying, node count is nearly free. This one first
appeared to be +10% — that was draft acceptance moving from 68% to 78%, not the change.

**Sparse attention on the draft's own context** (`LLAMA_MTP_QSA_MIN_T=1`). Slower. A single layer's
indexer costs more than the 0.8 ms dense read it replaces.

**Folding the catch-up into the first draft step.** No time saved, perturbs the drafts, and a
deferred stash breaks the batch position check when the cache is cleared.

**FR-Spec vocabulary pruning** (restrict the draft head to the frequent 64K). Chinese token ids sit
at 95K-131K in this vocabulary, so a 64K set covers only 84-96% of held-out text. The head is
bandwidth-bound, so a pruned head is genuinely cheaper — but not on text this model is used for.

## Kernels and quantisation

**A counting sort for expert ordering.** Reported elsewhere as +4.6-8.1%; here it lost ~4% (904/906
against 946). The atomic per-expert cursor serialises the scatter, and a 512-wide scan costs more
than a 1024-element all-LDS bitonic sort with no atomics.

**Elementwise chain fusion.** Correct and bit-identical (self-check finds 0 mismatches; 600 greedy
tokens identical), and worth at most ~1 ms/token — inside the decode probe's noise. It also loses to
the specialised binary kernels on prefill-sized tensors, so it is bounded to 262144 output elements.
Kept, because it is free and correct, but it is not a win.

Two traps it produced are worth keeping:

- **Folding a REPEAT that runs *before* the chain into a read of its source is a use-after-free** in
  allocator terms: the source's lifetime ends at the REPEAT node and the allocator hands that memory
  on. Only fold REPEATs inside the chain.
- The verification harness lied twice. A null-stream copy is not ordered against the non-blocking
  compute stream; and recomputing a chain "plain" *after* the fused write is invalid when the output
  is allocated in place over an operand. Snapshot operands, run plain first, restore, then fused.

**Requantising the trunk to IQ4_NL.** +10-12% perplexity. Double quantization: IQ3_S at 3.4 bpw
cannot be restored at 4.5 bpw. The same experiment on the whole model proved the prefill hypothesis
— an all-IQ4_NL file reached 1020 t/s against 903 — and must not ship for the same reason.

**Q6_K trunk, as a file or as in-memory twins.** A real trade-off, not a win: perplexity-neutral,
decode -7%, prefill -6%. The in-memory version costs +2.9 GB of commit, which at ctx 262144 / ubatch
8192 is enough to push production over: prefill fell to 816 t/s and a multi-turn chat died at turn 2
with `bad allocation`. Shipped inert, usable at ctx ≤ 131072, default off. The better design is a
native Q6_K dequant path in the matrix-core kernel rather than twins.

## Configuration

**A 64 GB carve.** Worse on both axes than 96 GB and now dead: the model does not fit, ~9.8 GB spills
into shared memory, and the driver pages during decode. 839 vs 942 t/s prefill, 49.6 vs 38.6 ms/token
decode, 17-19 vs 26-30 tok/s multi-turn — measured with the 64 GB arm at the *higher* power limit.
The user's own 84K chat: 96 GB 854.0 t/s / 58.5 ms against 64 GB 752.6 / 82.1.

**`parallel > 1` as a default.** Costs ~12 GB: the worst-case graph reserve uses a mixed-sequence
ubatch, which fails the sparse path's single-sequence visibility test and reserves the dense
per-block bias and the dense mask. Default is back to 1. (It remains free throughput for genuinely
concurrent users — see [decode-budget.md](decode-budget.md).)

**`-lzm on`** (mmap-style paging for the PLE table) is 4.2× slower than `-lzm on-direct` here.

**`QSA3_FORCE=1`** is a null test, not a shortcut. It lifts the kernel's `< 128 queries` gate, but the
packed K/V operands it needs are only built under the prefill layout, so the predicate declines
anyway. Perplexity, equivalence and speed all came back identical. Feeding that kernel at decode
would mean maintaining a packed copy of the whole cache incrementally: +6.4 GB at ctx 262144.
Gathering the selected cells for the ordinary attention kernel is what worked instead.

## Upstream

**Moving past the pinned revision.** Upstream HEAD, ten commits later and none of them benchmarked
upstream, measured 752.0 t/s against 810.8 on the same config — and does not compile on Windows. The
pin is deliberate.
