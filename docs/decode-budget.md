# Where a decoded token's time goes

At 85K context with speculation on, measured with `STRIX_SPEC_TIMING=1` and `STRIX_DECODE_TIMING=1`
(`tools/decode_lab.py --config dectime`). Numbers are milliseconds per speculative pass, which
produces 2.7-2.9 accepted tokens.

```
pass 86 ms
├── target verify         69     one batch of 1+3 tokens through the full model
│   ├── host              10.3   prep 1.5 │ graph 1.8 │ set_inputs 1.3 │ submit 5.7
│   └── device           ~58     of which ~43 is weight bandwidth
├── 3 draft steps         13     ~4.3 ms each: LM head 2.4 │ 85K draft KV read 0.8 │ MTP layer 0.65
├── catch-up               1
├── sampling               0.5
└── other                  1.6
```

Two facts that shape everything else:

**~43 ms of the 69 is reading weights.** A mixture-of-experts forward pass at this size moves 1.16 GB
of expert weights per token. That is not a kernel problem and does not move without changing the
model file.

**The host's 4.6 ms ahead of the submit does not overlap the GPU.** It is ~5% of a pass, and it is
the part that responds to ordinary optimisation.

## Why three draft tokens and not five

A MoE model breaks the usual assumption that batching is free. With 512 experts and 10 active per
token, a verification batch of n tokens touches more experts than a batch of one, so the verify cost
genuinely scales with the batch — unlike a dense model, where a 4-token verify costs about what a
1-token decode costs.

Measured marginal cost of one more draft position:

```
18.5 ms = 7.4 (one more draft step) + 11.2 (one more token of expert bandwidth in the verify)
```

A pass is ~28 ms per accepted token, so a position pays for itself only if it returns **0.64 extra
accepted tokens per pass**. Position 4 returns 0.56. Hence:

| `draft_max` | ms/token |
|---|---|
| 3 | **27.9** |
| 4 | 28.7 |
| 5 | 30.2 |

This ratio is a property of the model and the machine, not a tuning preference. Re-derive it rather
than copying the number if either changes: run the pass timing at each `draft_max` and compare
marginal cost against marginal acceptance.

## The draft's own cost

A draft step is ~4.3 ms, and **60% of it is the LM head** — one 2560 × 248320 projection, entirely
bandwidth-bound. That is why the draft gets its own quantised output projection
(`tools/make_draft_head.py` builds one): every draft token is verified by the target anyway, so the
head's precision affects acceptance and nothing else, and at IQ4_XS acceptance did not move. Worth
+5% on English and +9% on Chinese.

For the same reason a Q4_K_M draft body beats Q8_0 by under 1% and saves 880 MB — take the memory.

## Known slack

- **Graph reuse on speculative decodes is ~45%.** The draft context alternates between `1+n_draft`
  and `1` token shapes against a single cached graph result, so each shape evicts the other's. More
  than one live result needs scheduler surgery. (Folding the *shape* into the HIP graph cache key was
  a separate bug and is fixed — it was worth +2%, and it is why all the draft graphs replay at all.)
- **`set_input_kq_mask` scans all 85K cells once per decode**, about 0.9 ms. Rows after the first are
  already optimised upstream.
- **Concurrency.** Per-step time only goes 50.2 → 79.3 ms for four times the work, because weight
  bytes are read once per batch. Four streams give 47.1 tok/s aggregate against 19.0 for one. The
  default is `-np 1` because slots cost ~12 GB, but the machine is not compute-bound at decode — a
  GPU reading 75% busy during decode is memory stall plus the host gap between tokens.
  **Measured again with MTP on (2026-09-21):** 37.5 → 48.9 → 57.7 tok/s for 1 / 2 / 4 streams,
  i.e. 1.54× where speculation-off gave 2.5×. MTP had already spent most of the lever — one stream
  verifies 4 tokens per pass, four streams verify 16, and a 16-token pass costs 2.6× a 4-token one
  because the routed experts a batch touches grow with it; only the trunk, attention and the shared
  expert are read once. Bandwidth does not grow with slots; tokens per byte do, until the expert
  set saturates. The step fits `~30 ms + ~1 ms × distinct experts touched`, with 512 experts and
  10 per token: 10 / 39 / 76 / 139 / 238 experts for 1 / 4 / 8 / 16 / 32 tokens. Two consequences,
  both measured at eight slots: MTP saturates at four streams (62.4 → 62.9 tok/s from 4 to 8), and
  since its draft tokens are only 60–70% useful, **past four users speculation off is faster** —
  71.4 tok/s at eight streams without MTP against 62.9 with, 8.9 tok/s per user. Per-kernel measurement
  then showed the expert kernels near the ceiling in the one-user configuration (the ~110 GB/s
  read was every per-token cost charged to the experts); what was slow was IQ3_S in MMQ — byte
  loops in the tile loader on HIP — and a tiling cliff above 16 tokens per step, both fixed by
  `apply_iq3s_mmq_hip` (eight users with MTP 60.5 → 70.1 tok/s; one user unchanged). The
  per-token costs that do grow with users are the GDN recurrent state (fp32, ~3 MB per sequence per
  layer, read and written every step: 15 ms of a 112 ms eight-user step), the PLE gather and the
  per-token projections. Also: `-np 4` does not load at context 262144 (sticky HIP launch
  failure during the graph reserve; 131072 loads), and its mixed-sequence ubatches take the dense
  attention path, so the multi-stream QSA work this would need is worth at most that 1.5–1.9×.
