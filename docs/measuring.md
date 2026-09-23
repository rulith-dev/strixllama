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

## An expert kernel needs the model's routing

test-backend-ops draws expert ids uniformly, so at 8192 tokens every one of 512 experts gets ~160
rows. The model's routing is nothing like that: the median expert gets 57-82 rows, 20-31% get 16 or
fewer, 1.7-1.8% over 1024. Anything that decides by an expert's row count - tile widths, the big/small
threshold - comes out differently on the two. Record the routing of a real prompt with
`STRIX_MMB_GLU_DUMP=<file>` (one line per call, rows per expert; it synchronises, so only for recording)
and replay it in the benchmark with `STRIX_MOE_IDS_FILE=<file>` and `STRIX_MOE_GLU_PERF=8192`. A kernel
run also moves by up to 10% between runs on this machine: interleave the builds and take the minimum of
several.

## A mixed step's probabilities are not reproducible

When several conversations decode at once, which requests land in which batch depends on timing, and the
batch shape changes the numerics. The top-5 probabilities of the same token in a two- or four-conversation
step moved between two runs of one build (0.1-0.3 nats on the minor candidates). Compare greedy tokens
there; compare probabilities only with one conversation decoding.

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

## Which HIP runtime a build loads

Windows looks for a DLL in the executable's directory, then System32, and only then PATH. The display driver
installs its own `amdhip64_7.dll` and `amd_comgr` in System32, so a build with no ROCm DLLs beside it - the
build directory, and `bin/hip-rocm101` before 2026-09-24 - ran on the driver's HIP runtime, whatever PATH
said, while hipBLAS, rocBLAS and hipBLASLt (not in System32) came from the SDK. The bundle carries the SDK's
runtime beside the server. Same binaries, same prompt, the two layouts:

| | build layout (driver runtime) | bundle (SDK runtime) |
|---|---|---|
| 23K conversation, MTP | 29.50 tok/s, acceptance 147 / 304 | 37.77 tok/s, 170 / 240 |
| 23K conversation, no MTP | 44.78 ms/token | 43.47 ms/token |
| 18.6K prompt, the first token's top logprob, ubatch 8192 | -0.035 | -0.049 |
| the same at ubatch 4096 | -0.047 | -0.520 |

Every earlier comparison had both sides in the build layout, so they agreed with each other. It surfaced as
a "regression" of a build against the released 0.1.10 bundle, and swapping the build's binaries into a copy
of that bundle made it disappear. `bootstrap.py --build` now puts the SDK's copies of the DLLs System32
would shadow beside the build, and the two layouts agree bitwise. When in doubt, ask a running server what
it loaded:

```powershell
(Get-Process llama-server).Modules | Where-Object ModuleName -match 'amdhip|comgr' | Select-Object FileName
```

## "bad allocation" was the commit limit

The server turns `std::bad_alloc` into "decode() failed: bad allocation", and it came and went: once in
~16 multi-slot runs, then every time once a change made the server use a little more memory. A new handler
that prints the requested size settled it at once: 3-9 MB requests were failing, so the machine was out of
commit, not computing a garbage size. The commit limit is RAM plus page file, 31.6 + 96 = 127.6 GB, and on
this machine device allocations of ~3.5 GB and up count against it one for one. Two things follow:

- Watch commit, not RAM: `GlobalMemoryStatusEx` (`ullAvailPageFile`) and the server's `PrivateUsage`,
  sampled per phase of a run (load, each prefill, the decode). Working set and "Dedicated Usage" say nothing.
- Freeing does not give it back. ROCm on Windows keeps a freed buffer's commit, and a request a few MB
  larger never reuses it - a `hipMalloc`/`hipFree` loop through ctypes shows it in a minute. ggml-alloc
  reallocated the 3.4 GB compute buffer whenever a graph needed a few MB more, so each of those cost 3.4 GB
  for the rest of the process (`GGML_ALLOC_DEBUG=1` prints every reallocation).

## When a process dies with no message

Read the Windows Application Error event log immediately: it gives the faulting module and offset.
An access violation inside `memset` with no output cost an evening and four wrong hypotheses; the
event log settled it in one look. (The cause: an incremental graph stopped reading an input, so it
was never allocated, but `set_input` still filled it. Treat any graph input that some variant stops
reading as optional.)
