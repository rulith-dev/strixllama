# The patch set, and how it was audited

*This is the record of an audit run against the predecessor repository, kept because it explains why
the patch order is what it is and why five of these scripts were generated rather than written. The
tooling it names now lives in `bootstrap/bootstrap.py`.*

Audit of 2026-09-19, produced by `bootstrap/bootstrap.py --verify` and re-runnable at any
time. This exists because the repository accumulated 47 patch scripts across two backends, of which
only two were referenced from the manager or the scripts, and nothing recorded which ones produce the
binary that runs. Everything below is measured from the tree and the running process, not read off
documentation — one documentation claim turned out to be wrong (see IQ3_S).

## Upstream pin

    https://github.com/pwilkin/llama.cpp @ f5daaa3cfa6358e5dd398911ec741813745a5440

`bootstrap/UPSTREAM.json` carries the delta's provenance: for each of the 20 modified files, the
Git blob SHA-1 and SHA-256 it has upstream; for all 24, the SHA-256 the patch set produces. Those
two halves are what make the delta below provable rather than remembered — the first says what the
patches started from, the second what they must end at.

## The whole delta: 24 files

20 modified, 4 added, out of 3610.

| Area | Files |
| --- | --- |
| CUDA/HIP backend | `ggml-cuda.cu`, `mmb.cu`, `mmb.cuh`, `mmid.cu`, `mmvq.cu`, `softcap.cu`, `softcap.cuh`, `vecdotq.cuh` |
| added CUDA/HIP | `strixllama-chain.cu/.cuh`, `strixllama-getrows-cast.cu/.cuh` |
| llama core | `llama-context.cpp`, `llama-graph.cpp`, `llama-lazy-reader.h`, `llama-memory-hybrid-idx.cpp/.h`, `llama-memory-recurrent.cpp/.h`, `llama-model.cpp/.h` |
| model | `models/qwen4exp.cpp` |
| speculation / server | `common/speculative.cpp`, `tools/server/server-context.cpp` |

## The 17 patch scripts that are live

Verified by running each against a throwaway copy of the four source directories the patches touch
and keeping those that report "already applied". Each script gets a fresh copy, so one cannot mask
another.

    apply_chain_fusion            apply_qsa_block_key_cache
    apply_decode_timing           apply_qsa_decode_gather
    apply_getrows_cast_fusion     apply_qsa_kb_image_guard
    apply_graph_key_shape         apply_qsa_small_batch_mask
    apply_hip_ple_probe           apply_rs_pos_warn_stateless
    apply_iq3s_vecdot_hip         apply_skip_ops_ablation
    apply_mmvq_rdna35_rows        apply_spec_draft_ubatch
    apply_ple_overlapped_reads    apply_spec_timing
    apply_trunk_decode_twins

Six files are written by more than one script, so order matters when replaying onto a clean tree:

| File | Scripts, in the order they must run |
| --- | --- |
| `ggml-cuda.cu` | chain_fusion, getrows_cast_fusion, graph_key_shape, skip_ops_ablation, hip_ple_probe |
| `models/qwen4exp.cpp` | qsa_block_key_cache, qsa_decode_gather, qsa_small_batch_mask, qsa_kb_image_guard, hip_ple_probe |
| `llama-memory-hybrid-idx.cpp/.h` | qsa_block_key_cache, qsa_kb_image_guard |
| `strixllama-chain.cu/.cuh` | chain_fusion, getrows_cast_fusion |

Of the other 30 scripts, 29 do not apply to this fork at all — they target the Vulkan backend or
files this tree does not have — and one (`apply_mtp_pending_reset`) applies cleanly, meaning it is
not in the build.

## The gap: 5 modified files no script owns

`mmb.cu`, `mmb.cuh`, `softcap.cu`, `softcap.cuh`, `mmid.cu`.

**`mmid.cu` is a comment only**: a note recording that a counting sort was tried in place of the
bitonic expert sort and lost (906 against 946 t/s pp16384). No code change.

**The other four are the IQ3_S MMB kernel and the sigmoid fusion, and they are live in production.**
They exist only as whole-file copies in `patches/iq3s-kernel/`, with no `apply_*.py` script, so a
clean rebuild from the patch scripts alone would silently omit the single largest measured win in
the project: +4.8% prefill and +5.5% decode (pp16384 947.37 against 903.64, tg128 24.97 against
23.66), from letting the 52.2% of the model body stored as IQ3_S reach the matrix cores.

`patches/iq3s-kernel/README.md` states "It is not in production". That is wrong, on three
independent pieces of evidence:

1. The four files in that directory are byte-identical to the production tree.
2. `mmb.cu:921` passes `allow_iq3s = true` from the fused-GLU caller, with no runtime gate.
3. The running server logs `MMB_GLU fused gate/up+swiglu: ... type=IQ3_S`.

### Settled 2026-09-19: the rollback reason is obsolete, keep the kernel

The rollback was attributed by elimination, not root cause, and predates the real fix. On 2026-09-18
every "answer stops mid-sentence" report was traced to speculative verification batches running dense
attention with no causal mask, fixed by `apply_qsa_small_batch_mask.py`, which is live.

Re-measured on the current build. `STRIX_MMB_IQ3S=0` was added to `mmb.cu` so both arms are the
same binary with one variable; `tools/truncation_test.py --fill 70000 --reps 1`, fresh server per
arm, verified by `type=IQ3_S` appearing in one server log and not the other:

| | 68499-token prefill | answer | sections |
| --- | --- | --- | --- |
| IQ3_S on | **876.7 t/s** | 1110 tokens, `finish_reason=stop` | **30/30** |
| IQ3_S off | 830.6 t/s | 1085 tokens, `finish_reason=stop` | **30/30** |

Both reach 30/30: the truncation is gone. Prefill is **+5.6%**, matching the +4.8% originally
claimed. The kernel stays, and `STRIX_MMB_IQ3S=0` stays as the switch to reach for when a future
numerics or long-generation regression has to be attributed.

(The decode rates in that test are not a comparison - it is a speculative run and the two arms drew
different draft counts.)

## The deeper gap: the scripts cannot rebuild the tree on their own

`ggml-cuda.cu` also carries unscripted content, and the counts line up exactly with the snapshot in
`patches/iq3s-kernel/`, not with anything the live scripts insert:

| marker in `ggml-cuda.cu` | production tree | all 17 live scripts | iq3s snapshot |
| --- | --- | --- | --- |
| `GGML_UNARY_OP_SIGMOID` | 11 | 0 | 11 |
| `scale_sigmoid` | 2 | 0 | 2 |
| `NO_SIGMOID_FUSE` | 2 | 0 | 2 |
| `LLAMA_GRAPH_DUMP` | 3 | 0 | 3 |
| `GT compute` | 1 | 0 | 1 |
| `mmb_supported_mmid` | 3 | 0 | 3 |

So the production tree is **snapshot first, scripts second** - the 17 scripts layer onto
`patches/iq3s-kernel/`, not onto clean upstream. `GT compute` is the `LLAMA_GRAPH_TIMING` line the
whole decode-budget analysis is built on, and it exists only in that snapshot.

A bootstrap therefore has three steps, and this order is not optional:

1. check out `pwilkin/llama.cpp` at `f5daaa3...`;
2. overlay the six files in `patches/iq3s-kernel/` (whole files, hash-verifiable);
3. apply the 17 live scripts, respecting the ordering table above.

## Replay result, 2026-09-19: 24 of 24 — the bootstrap is proven

    clean-upstream check: ok
    patches   : 21 applied, 0 failed
    REPRODUCED 24 / 24

Every file of the delta is byte-identical to the production tree, starting from a clean upstream
reconstructed from hash-verified downloads. `bootstrap/bootstrap.py --verify` re-runs it in a few minutes.

Getting there needed one ordering fix and five new scripts:

- **`apply_win_lazy_reader` must run first.** It installs the cross-platform lazy reader that later
  patches anchor against. It had looked dead because it was being run after the snapshot handed it
  its own finished output.
- **The snapshot must not supply `llama-lazy-reader.h`.** Same trap from the other side: give
  `apply_ple_overlapped_reads` the finished file and it reports "already applied" and skips the half
  of its work that lives in `qwen4exp.cpp`.
- Five scripts generated by `tools/make_patch_script.py` from the tree itself, with anchors grown
  until unique, replacing what had drifted or was never captured:

| New script | Closes |
| --- | --- |
| `apply_node_timing` | `STRIX_NODE_TIMING`, which existed only as an uncommitted edit |
| `apply_iq3s_vecdot` | supersedes `apply_iq3s_vecdot_hip`, whose anchors matched nothing |
| `apply_trunk_twins` | supersedes `apply_trunk_decode_twins`, written against a modified tree |
| `apply_ple_prefetch_launch` | the PLE prefetch half no script performed |
| `apply_mmid_sort_note` | the comment recording why the expert sort stays bitonic |

All five are idempotent, report "already applied" against the production tree, and modify nothing
when re-run.

### The audit that got here (kept for the reasoning)

`tools/replay_bootstrap.py` does exactly that and is re-runnable. Only 20 files differ from
upstream, so only those are restored: normally out of the clone's own git objects, addressed by the
blob hash in `bootstrap/UPSTREAM.json`, or with `--fetch` through a mirror and checked against the
recorded SHA-256 — all 20 verified either way. The reconstructed tree then matches upstream in every recorded file,
the snapshot goes on, the scripts run in the order above, and every delta file is compared.

    clean-upstream check: ok
    patches   : 15 applied, 2 failed
    REPRODUCED 17 / 24

So most of the build is now provably reproducible, and the seven that are not each have a named
cause. None of them is guesswork:

| File(s) | Why it does not reproduce |
| --- | --- |
| `ggml-cuda.cu` | `STRIX_NODE_TIMING`, the per-dispatch HIP event instrumentation, is in **no script and not in the snapshot** — an unrecorded edit straight into the tree |
| `mmid.cu` | comment-only note about a rejected counting sort; unrecorded, harmless, fold into the snapshot |
| `vecdotq.cuh` | `apply_iq3s_vecdot_hip` is stale: its `OLD_LOOP` anchor occurs in neither upstream nor the production tree, so the script can no longer produce what the tree contains |
| `models/qwen4exp.cpp` | the PLE prefetch half (`launch_prefetch` / detached thread) has no working script; `apply_hip_ple_probe` reports "already applied" and writes nothing |
| `llama-model.cpp`, `llama-model.h`, `llama-graph.cpp` | `apply_trunk_decode_twins` fails with "anchor not unique" against clean upstream — written against an already-modified tree |

All five were closed the same day; see the result at the top. Had they not been, a clean rebuild
would have silently omitted the node timing, the IQ3_S vec-dot, the PLE prefetch and the trunk
decode twins — which is the whole reason this audit exists.

    python bootstrap/bootstrap.py --fetch --patch    # download upstream once, then replay
    python tools/replay_bootstrap.py            # replay again offline

## How to re-check

    python bootstrap/bootstrap.py --record     # hash what the patch set produces
    python bootstrap/bootstrap.py --verify     # is this tree still exactly that?

A file that appears under `--delta` and is named by no live script is a reproducibility gap: the
build cannot be recreated from the patch set alone.

## Addendum 2026-09-21: `apply_iq3s_mmq_hip`

The delta is now 27 files (23 modified, 4 added); the replay reports 27 / 27. One script, four files,
last in the order because it rewrites the sign table `apply_iq3s_vecdot` added:

| file | what |
|---|---|
| `ggml/src/ggml-cuda/vecdotq.cuh` | the IQ3_S sign-byte mask becomes two multiplies (`strixllama_iq3s_sign_mask(s4)`) instead of a 16-entry device table; the mmvq vec-dot uses it |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | the IQ3_S MMQ tile loader on HIP: no `__vcmpne4`/`__vsub4` byte loops, grid read from the LDS copy |
| `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_process_tile` fills the LDS grid after the x tile (`mmq_get_nbytes_shared` reserves it); the fork's compact routed MoE path admits IQ3_S and IQ4_XS, not only IQ4_NL |
| `tests/test-backend-ops.cpp` | `STRIX_MOE_PERF` perf cases for this model's expert shapes (512 experts, 10 used, 640x2560 / 2560x640) |

Measured with `test-backend-ops perf -o MUL_MAT_ID` under `STRIX_MOE_PERF`: IQ3_S gate/up 123 -> 149 GB/s
at 8 tokens per step, 117 -> 157 at 16, 73 -> 156 at 32; IQ4_XS at 32 tokens 74 -> 170. See
`docs/results/concurrency-mtp-20260921.json` for the decode-level effect.

`tools/make_patch_script.py` learned to merge hunks whose anchors overlap after an earlier hunk has
been applied, and to replay its own output against the before-tree before writing it - the first
version of this script had four anchors in one loop body, of which the second no longer matched once
the first had been applied.

## Addendum 2026-09-21: `apply_prompt_cache_disk`

The delta is now 29 files (25 modified, 4 added); the replay reports 29 / 29. Three files, last in
the order because its server-loop anchors sit in text earlier scripts wrote:

| file | what |
|---|---|
| `tools/server/server-task.h` | `server_prompt_cache` gains a disk tier: an index of on-disk entries, `set_disk`, `persist`, `load_from_disk` |
| `tools/server/server-task.cpp` | the file format (tokens, target and draft state, checkpoints), writing on every RAM-cache insert, prefix-matched reads on a RAM miss, size-bounded eviction oldest first |
| `tools/server/server-context.cpp` | `STRIX_PROMPT_CACHE_DIR` / `STRIX_PROMPT_CACHE_MIB` wire it up after the RAM cache is created; `prompt_save` persists what it just captured |

Why it is a tier under the RAM cache and not a slot-file feature: the slot save/restore endpoints
carry the state but not the recurrent checkpoints, and without a checkpoint the server has to
re-process a hybrid model's prompt from the start ("forcing full prompt re-processing due to lack of
cache data") - measured: restore in 0.5 s, then 38 s of prefill anyway. The RAM cache entry is the
complete unit, so that is what goes to disk.


## Addendum 2026-09-21: `apply_multi_stream_qsa`

The delta is now 30 files (26 modified, 4 added); the replay reports 30 / 30. Four files, 38 hunks,
last in the order because its anchors sit in text the earlier scripts wrote:

| file | what |
|---|---|
| `src/llama-memory-hybrid-idx.h` | `qsa_mixed_inputs` (two membership inputs) threaded through `set_input_qsa_blocks` |
| `src/llama-memory-hybrid-idx.cpp` | the block-key cache gap check per token and per sequence; `qsa_scalar_visibility` accepts several sequences; the compact fill writes `seq_blk` / `seq_tok` |
| `src/models/block-graph.inc` | `qwen4exp_apply_compact_visibility` folds `seq_blk^T seq_tok` (exactly 0/1) into the visibility |
| `src/models/qwen4exp.cpp` | the decode gather batched on the flash-attention sequence axis with f16 rows (limit 32 queries); the membership inputs built, set and checked for reuse; `dirty_max` grows with the sequence count |

Why: with more than one slot the server's equal-length split puts several sequences in one ubatch.
The sparse path declined those, so they ran dense attention over the whole pool (four users 35K
deep: 28 tok/s together against 32 for one alone), a multi-stream step tripped the sticky block-key
cache gap flag for every stream, and the mixed reserve graph kept a dense f16 mask of n_kv × ubatch
that the driver would not fill at 262144 × 8192. Measured in `docs/results/concurrency-mtp-20260921.json`
(`multi_stream_qsa_20260921`): four users 35K deep 33–39 tok/s, four slots at 262144 × 8192 load
and cost 1.3 GB over one. `src/models/block-graph.inc` joins the upstream side of the delta
(`bootstrap/UPSTREAM.json`).

## Addendum 2026-09-21: `apply_small_m_mmvf`

Three files already in the delta (still 30 files, replay 30 / 30), 10 hunks:

| file | what |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | a few-row F32/F16 src0 (≤ 64 rows) with 9–64 columns runs the vector kernel over column chunks of 8 instead of falling through to cuBLAS |
| `tools/server/server-context.cpp` | `STRIX_SPEC_TIMING` splits "other" into pre / post / gap |
| `tests/test-backend-ops.cpp` | `STRIX_DENSE_PERF` (the big Q8_0 projections, 8 distinct matrices so the working set beats the MALL) and `STRIX_SMALL_PERF` (the few-row shapes) |

Why: a 16-token verify step (four slots) ran the hyper-connection inject `[10240 × 4]` at 117 µs
and the GDN beta/alpha `[2560 × 48]` at 37 µs per call through cuBLAS — 95 + 72 calls per pass —
because `mul_mat_f` declines a src0 narrower than its row tile and the vector kernel stops at 8
columns. With chunks: 13 and 10 µs; the bare 16-token pass 139.8 → 130.4 ms. Measured in
`docs/results/concurrency-mtp-20260921.json` (`pass_budget_20260921`).
