# Measuring this thing without fooling yourself

Most of the wrong conclusions in this project came from measurement, not from code. They are written
down here because every one of them cost hours and every one of them looks reasonable at the time.

## Wall-clock tok/s is not comparable between speculative runs

This is the big one. With speculation on, throughput is `pass_time / tokens_accepted_per_pass`, and
**any** numerics change reshuffles which drafts the target accepts. A change that moved acceptance
from 68% to 78% on one prompt looked like +10% throughput and was, on a clean same-binary A/B,
exactly neutral: 83.8 ± 6.0 ms per pass against 83.6 ± 5.7.

So compare *pass time at a fixed batch size*, not tok/s:

```bash
python tools/decode_lab.py --config spectime     # STRIX_SPEC_TIMING=1
```

which prints `ST pass=… n=4 draft=… decode=… process=… sample=… other=…` per speculative pass.
Run-to-run sigma on `pass` is ~6-8 ms, so **a single run cannot resolve anything under ~3 ms**. If
the effect you are chasing is smaller than that, you need either many runs or a different probe.

When you do report tok/s, report acceptance beside it. A tok/s figure without acceptance is not a
measurement of the runtime, it is a measurement of the prompt.

## Acceptance needs real text

The depth probe generates random words. They are undraftable, so acceptance goes to noise and any
speculation number taken from them is meaningless. Use `--gen-prefix` with a real document for
anything acceptance-sensitive. Note that `--gen-prefix-chars` counts characters: ask `POST /tokenize`
what that is in tokens rather than assuming a ratio, which varies by a factor of four across
languages.

## Perplexity at ctx 2048 cannot see a deep-context bug

Two separate regressions passed a ctx-2048 perplexity gate cleanly and were caught only by generating
long answers:

- the maskless verification leak read 2.3807 at ctx 2048, and 1.0841 at ctx 4096
- the IQ3_S kernel was blamed for truncation on a gate that reported 2.3786 in every variant

Sparse attention only engages above ~2K context, so a 2048-token gate is testing a different code
path from the one that runs. **Any numerics change has to pass perplexity at a context where sparse
attention is active, at small and large ubatch, and `tools/truncation_test.py --fill`** — 30 numbered
sections at temperature 0, which fails visibly when an answer stops early.

## `llama_decode` is asynchronous

It returns once the work is submitted. So the "decode" figure in the speculation timing is mostly
`llama_synchronize` waiting for the GPU, not decode. To split host from device use:

```bash
python tools/decode_lab.py --config dectime      # adds STRIX_DECODE_TIMING=1
```

which breaks a decode into `prep / graph / inputs / compute / post` and reports how often the graph
was reused.

## When an ablation cannot find a cost, time the nodes

Skipping whole op classes (`--config skip-sel`) left the long-context decode slope at 0.31 ms per
1000 tokens and pointed at nothing, because the cost was hiding in generic ops — ROPE, GET_ROWS and
RMS_NORM over the whole cache, not in the attention op. A per-dispatch timing run named it in one
pass:

```bash
python tools/decode_lab.py --config nodetime     # STRIX_NODE_TIMING=1, graphs disabled
```

Node timing needs HIP graphs off, so its absolute numbers are not production's. Use it to find
*which* node, then measure the fix normally.

## Node count is nearly free

HIP graphs replay node-by-node on the host, so removing nodes from a replayed graph buys very little.
Batching a per-query gather removed ~900 nodes from a 4-token graph and measured exactly neutral.
Stop treating dispatch count as a cost once graphs are replaying; measure it.

## Things that quietly invalidate a run

- **A server that has served an image** decodes ~7% slower at long context for the rest of that
  slot's life, because the block-key cache is switched off for any memory that has seen one and only
  a full clear turns it back on. Acceptance is unchanged, so nothing in the speculation counters
  hints at it. Restart the server before a decode measurement, or the number describes the chat
  history rather than the build. This produced a 32.1 ms/token reading against a true 28.7.
- **Benchmarking straight after a compile** measures a cold page cache for the PLE table. Prefill
  swings four-fold on that alone — 852 warm against 212 cold, same binary, same flags. This was once
  misreported as a 76% regression.
- **Another GPU process still shutting down.** Leave ~30 s between runs; a perplexity run started
  immediately after another hit a transient device OOM.
- **CPU load from a build or a requant.** An A/B that declared a decode kernel slower was taken under
  a concurrent build; rerun clean, it was +2%.
- **Thermals after a concurrency test.** A 15 tok/s reading a minute after a four-stream burst was
  heat, not the change under test.
- **Comparing a 3-rep run against a 1-rep baseline.** This produced a confident "+9%" that did not
  exist.
- **Not checking which binary actually ran.** A script that rewrote a temp file never substituted the
  runtime path, so three "new build" runs were the old binary compared against itself. Print the
  resolved runtime on every run.
- **Not checking which model actually ran.** `decode_lab.py` without `--model` takes the catalog's
  first model, which is whichever sorts first on the machine: with Qwen3.6-35B-A3B on disk, a
  "Flash-Next prefill profile" was of that model, dense attention and all. Pass `--model` and read
  the `model:` line it now prints.
- **`llama-bench` has no `--load-mode none`**, so at this carve it maps 93.7 GB through a 31.6 GB
  page cache and thrashes the page file. Measure through the server. Its `tg128` also says nothing
  about real chat throughput on this model: ~25 against 64.8 measured on a coding turn.

## GPU "Dedicated Usage" is not an allocation figure

Under `--load-mode none --lazy-mode on-direct`, weights page in on demand, so the counter tracks how
much context has been processed — 60 GB after a 2.3K prefill, 83 GB after 85K, same model. Only
compare it between runs of an identical workload.

## When a process dies with no message

Read the Windows Application Error event log immediately: it gives the faulting module and offset.
An access violation inside `memset` with no output cost an evening and four wrong hypotheses; the
event log settled it in one look. (The cause: an incremental graph stopped reading an input, so it
was never allocated, but `set_input` still filled it. Treat any graph input that some variant stops
reading as optional.)
