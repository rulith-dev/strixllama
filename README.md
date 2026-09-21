# Strix Llama

Serving a 125B mixture-of-experts model fast on one AMD Strix Halo machine, on Windows.

Target: **Ryzen AI Max+ 395** (Radeon 8060S, gfx1151, 128 GB unified memory) running
**Qwen3.8-Flash-Next** (125B-A6B, Unsloth UD-IQ4_XS, 93.7 GB) through a patched llama.cpp.

*[中文说明](README.zh.md)*

## Where it stands

Measured on the target machine, 85K tokens of real prose, speculative decoding on:

| | |
| --- | --- |
| prefill | **886 t/s** |
| decode, 85K context | **28.7 ms/token** (34.9 tok/s at 68% draft acceptance) |
| decode, short context | **26.8 ms/token** (37.4 tok/s at 60% acceptance) |
| image input | supported (Qwen3-VL projector) |

For reference, the same model in LM Studio on the same machine decodes at about 18 tok/s.

Every number in this repository comes with the command that produced it, in
[docs/results.md](docs/results.md). Where a change could not be resolved above the noise floor, it
says so rather than claiming a win — and decode figures carry their draft acceptance, because with
speculation on, throughput without acceptance describes the prompt rather than the runtime.

## Just want to run it?

**[docs/getting-started.md](docs/getting-started.md)** — the installer from the
[releases page](https://github.com/rulith-dev/strixllama/releases), the five model files to download
and where to put them, and what to press. No Python, ROCm or build tools involved.

## Quick start (from source)

```bash
python bootstrap/bootstrap.py --toolchain          # ROCm SDK + ninja into toolchain/, ~5 GB
python bootstrap/bootstrap.py --fetch --patch --build
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
```

`bootstrap.py` clones `pwilkin/llama.cpp` at a pinned revision, applies the patch set, and builds
against the ROCm SDK. The same 30-file delta is also published as one commit on a fork, so it can be
read as a plain diff: [rulith-dev/llama.cpp, branch `strixllama`](https://github.com/rulith-dev/llama.cpp/tree/strixllama). `tools/manager.py` is a JSON-on-stdin process manager: it owns the launch
flags, the environment gates and the runtime, so a configuration is reproducible rather than
remembered.

**Read [docs/install.md](docs/install.md) first.** Three things there are not optional and not
obvious: the GPU carve must be 96 GB (at 64 GB the model does not fit and decode is 28% slower, which
no software setting recovers), the ROCm SDK must be TheRock 10.1 rather than the system 7.1 (+60%
prefill on identical source), and the SDK version this was measured on comes from a nightly index
with a ~27-day window — so the pin will stop resolving, and that page says what to do about it. The
model files are a separate download of about 95 GB.

Optional: `integrations/jan/apply.py` overlays three management pages into a
[Jan](https://github.com/menloresearch/jan) checkout, in English and Chinese.

## What is actually in here

The whole delta against upstream llama.cpp is **30 files** — 26 modified, 4 added, out of 3610. The
substantial pieces:

| | |
| --- | --- |
| **IQ3_S matrix-core path** | 52% of this model's body is stored as IQ3_S, which the MMB dequant GEMM did not accept, so half the weights never reached the matrix cores. +5.6% prefill. |
| **Sparse attention at decode** | The sparse kernel needs a packed layout only prefill builds, so a decoded token attended densely over the whole cache. Gathering the selected cells instead: 97K decode 59.0 → 49.3 ms/token, and the context slope drops from 0.188 to 0.067 ms per 1000 tokens. |
| **MTP speculation, tuned** | Draft head with its own IQ4_XS output projection, three draft tokens, n-gram drafting off. 85K decode 21.5 → 34.9 tok/s. |
| **Two correctness fixes** | Speculative verification batches ran dense attention with no causal mask, so long answers drifted and stopped early. Image input aborted the server three separate ways in the QSA block machinery. |
| **Measurement instrumentation** | Per-graph, per-dispatch and per-phase timing, all off unless an environment variable is set. |

`patches/MANIFEST.md` lists every patch, what it touches, and the order they must run in.

## How the build is verified

A patch set that cannot be replayed is a patch set that quietly rots. Two tools keep this honest:

```bash
python bootstrap/bootstrap.py --record    # hash what the patch set produces
python bootstrap/bootstrap.py --verify    # is this tree still exactly that?
```

`--verify` answers the weak question — is this tree what the recipe produced? It passes just as
happily if the tree was edited by hand and re-recorded afterwards. The strong question is whether the
recipe still rebuilds the tree from nothing, and that has its own tool:

```bash
python tools/replay_bootstrap.py          # clean upstream + patch set == the 30 files, byte for byte
```

It restores the 26 modified files to upstream from the clone's own git objects, addressed by the blob
hashes in `bootstrap/UPSTREAM.json` — so clean upstream is reconstructed rather than trusted — then
replays the snapshot and every script and compares. It reports **30 / 30**.

It was not always so. Five of the 24 were owned by no script at all - including the largest measured
win in the project, which a clean rebuild would have silently dropped - and three scripts had drifted
until they no longer applied. `tools/make_patch_script.py` generates a patch script from two trees,
growing each anchor until it is unique, which is how those were closed. Hand-written anchors are what
rotted in the first place.

## Scope and limits

- **Windows only.** The manager uses Win32 process APIs; the build targets gfx1151.
- **One model family.** The attention work is specific to `qwen4exp`. The kernel work (IQ3_S MMB,
  MMVQ rows-per-block) applies more broadly.
- **Not a llama.cpp fork.** This repository holds patches, tools and measurements. No model weights,
  no upstream source, no binaries.

## Layout

```
bootstrap/   fetch, patch and build the runtime; UPSTREAM.json is the delta's hash record
patches/     the patch set, its manifest, and two whole-file drops too large to anchor
tools/       manager, measurement harnesses, verification
integrations/jan/   optional management pages for a Jan checkout
docs/        install.md · results.md · measuring.md · decode-budget.md · dead-ends.md
```

[docs/measuring.md](docs/measuring.md) is the one to read before trusting any number here, including
the ones above — it is a list of the ways this project measured itself wrong.

## Licence

MIT, see [LICENSE](LICENSE). Third-party attribution in [NOTICE.md](NOTICE.md) — in particular this
patches [pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp) (MIT) and the Jan overlay is meant
for a [Jan](https://github.com/menloresearch/jan) checkout (Apache-2.0).
