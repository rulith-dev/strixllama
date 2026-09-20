# Installing and running it

**The short way** is the installer from the [releases page](https://github.com/rulith-dev/strixllama/releases):
`strixllama_<version>_x64-setup.exe` carries the desktop app and, under `runtime/`, everything it
runs — `llama-server` built from the pinned fork, the eight ROCm DLLs it imports plus the gfx1151
kernel libraries, the Visual C++ and OpenMP runtimes, the manager and an embedded Python. On a
machine set up as in section 1, with the model files from section 5 already in LM Studio's folder,
it needs nothing else: install, open, load the model from the Models page. The bundle is made by
`tools/make_runtime_bundle.py` from a finished build; `BUNDLE.json` inside it records what went in
and the upstream commit it came from.

The rest of this page is the long way, for building it yourself. Four things have to be in place:
the machine set up correctly, a toolchain, a build, and the model files. The build is the easy part.

## 1. The machine

| | |
|---|---|
| GPU | Radeon 8060S / **gfx1151** (Ryzen AI Max+ 395). Other RDNA 3.5 parts are untested; the kernels are compiled for this target only. |
| Memory | 128 GB unified |
| **GPU carve** | **96 GB**, set in the BIOS |
| OS | Windows 11 |
| Disk | ~110 GB for the model, ~15 GB for the toolchain, ~10 GB for the build |

**Set the carve before anything else.** At 64 GB the 93.7 GB model does not fit: about 9.8 GB spills
into shared memory and the driver pages during decode. Measured, 64 GB against 96 GB: prefill 839 vs
942 t/s, decode 49.6 vs 38.6 ms/token, multi-turn 17-19 vs 26-30 tok/s — and the 64 GB arm had the
*higher* power limit. No software setting recovers this. It leaves Windows 31.6 GB, which is enough.

Raising the carve lowers the commit limit; on this machine that is better solved with a fixed large
page file than by shrinking the carve back.

## 2. Software prerequisites

- **Python 3.12+**
- **CMake** and **git** on `PATH`
- **Visual Studio 2022** with the C++ workload. The ROCm SDK's clang still links against the MSVC
  runtime and uses the Windows SDK headers, so `vcvars64.bat` has to exist — `bootstrap.py` finds it
  and merges its environment, so you do not need a developer prompt.

## 3. The toolchain

```bash
python bootstrap/bootstrap.py --toolchain
```

This creates `toolchain/rocm-venv`, installs the ROCm SDK into it as wheels, expands the devel SDK
(`rocm-sdk init` — the wheel is a tarball until then, and a bootstrap that skipped this had no
compiler and no device bitcode), and downloads ninja. About 5 GB.

**Use TheRock ROCm 10.1, not the system ROCm 7.1.** Same source, same flags: 458 → 735 t/s, +60%.
This is the single largest factor in the prefill numbers and it is not a code change. Windows wheels
are the only form AMD ships — the native tarballs are Linux-only.

### The pin will expire, and you should plan for it

`ROCM_VERSION` is pinned to `10.1.0a20260910`, which is what every number in this repository was
measured on. It comes from a **nightly index with a rolling window of about 27 days**, and
`10.1.0a20260910` is the last nightly of the 10.1 series — 10.2 started the next day. So:

```bash
python bootstrap/bootstrap.py --toolchain --rocm-version <a version the index still has>
```

`--toolchain` prints the versions the index currently offers when the pin fails to resolve.

**Anything other than the pinned version is untested here.** A different SDK can change numerics, and
this project has twice shipped a regression that a short perplexity run could not see. If you move
off the pin, re-run the gates in [measuring.md](measuring.md) before trusting the build —
`tools/truncation_test.py --fill` at minimum.

If you have the pinned version working, **archive the wheels now**, because the index will not have
them in a month:

```bash
toolchain/rocm-venv/Scripts/python.exe -m pip download --pre \
  --index-url https://nightly.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,devel,device-gfx1151]==10.1.0a20260910" -d toolchain/rocm-wheels
```

and install from that directory afterwards with `--no-index --find-links toolchain/rocm-wheels`.

## 4. Build

```bash
python bootstrap/bootstrap.py --fetch --patch --build
```

- `--fetch` clones `pwilkin/llama.cpp` at the pinned revision into `src/llama.cpp`.
  The result of `--fetch --patch` is what the `strixllama` branch of [rulith-dev/llama.cpp](https://github.com/rulith-dev/llama.cpp/tree/strixllama) holds as a single commit, if you would rather read the delta than replay it.
- `--patch` overlays `patches/iq3s-kernel/` and runs the scripts in order.
- `--build` configures and compiles, then copies the runtime into `bin/hip-rocm101/`.

`--build` re-runs cmake every time on purpose: cmake resolves its `*.cu` glob at configure time, so a
file the patch set adds or renames is invisible to an existing `build.ninja`.

Check it against the recipe:

```bash
python tools/replay_bootstrap.py     # clean upstream + patch set == the 24 files. Expect 24 / 24.
```

## 5. Model files

Three files, none of them in this repository.

**The model** — Unsloth's `Qwen3.8-Flash-Next-GGUF`, quant `UD-IQ4_XS`, three shards, 93.7 GB. Point
the download at the first shard's siblings; the manager groups shards by name and refuses to start if
one is missing.

**The vision projector** — `mmproj-F16.gguf`, 904 MB, shipped in the same repository. Put it *beside*
the model: the manager looks for it there. Without it the server has no multimodal capability at all
and rejects any request carrying an image. It costs 904 MB of GPU memory and nothing else.

**The MTP draft** — start from Unsloth's `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`, then give it its
own output projection:

```bash
python tools/make_draft_head.py --type iq4_xs --base <the shared-Q4_K_M file> --target <shard holding output.weight>
```

The `shared-*` drafts borrow the target's Q6_K head, which streams 521 MB on every draft step — 2.4 ms
of a 4.3 ms step. Every drafted token is verified by the target, so a coarser head can only cost
acceptance, and at IQ4_XS acceptance did not move: worth +5% on English and +9% on Chinese.

### Already have them in LM Studio? Nothing to download

If LM Studio is installed, its download folder is picked up automatically: the manager reads
`downloadsFolder` from `~/.lmstudio/settings.json` and adds it as a model root. LM Studio lays
models out as `<root>/<publisher>/<repo>/<file>.gguf`, which is exactly the shape the catalog walks,
so everything already on disk shows up — **the same files, read in place, no import step and no
second 93.7 GB copy.** Both applications can keep using them; nothing is moved or rewritten.

If your models are somewhere else, add the directory to `roots` in `config/jan/settings.json`. Paths
outside a registered root are refused — that check is deliberate, since the manager runs model files
chosen over IPC.

## 6. Run

```bash
python tools/manager.py <<< '{"op":"catalog","data":{"refresh":true}}'   # find models, get their ids
python tools/manager.py <<< '{"op":"start","data":{"id":"<model-id>"}}'
python tools/manager.py <<< '{"op":"status"}'
python tools/manager.py <<< '{"op":"stop"}'
```

The server is then an ordinary OpenAI-compatible endpoint on `http://127.0.0.1:8080/v1`.

**The defaults are the measured configuration.** A first load runs with context 262144, batch and
ubatch 8192, flash attention on, sparse attention and MTP on for this model, one slot — the
settings every number in [results.md](results.md) was measured with — so nothing has to be set to
get the claimed performance. On a carve this does not fit, the load falls back to shared memory
(next section) rather than failing; a smaller context is the setting to change if that is too
slow. Sparse attention has to stay on above ~64K: dense attention runs out of memory there, and it
is slower below it.

### Shared GPU memory is decided per load, not by a switch

A load first runs with `GGML_HIP_ENABLE_UNIFIED_MEMORY=0`, everything in the dedicated carve: that
is 4% faster at prefill and, more to the point, steady — 903.6 ± 3.7 t/s against 867.1 ± 24.1 with
it on. If the load dies of out-of-memory there, the manager loads it again in shared memory and
remembers that for this model, this carve and these memory-relevant settings (`shared_vram_auto` in
`config/jan/settings.json`), so the next load goes straight to what works. Enlarge the carve or
change the context, batch, slots or draft and it is tried in the carve again. A profile can still
force shared memory with `"shared_vram": true`; the Configuration page has no control for it and
only says something when a load had to fall back. The dedicated size is read from the display driver's registry
entry, so it costs nothing and needs no GPU context.

### Thinking depth

`thinking` takes `off`, `low`, `medium` or `high`, and the Configuration page offers the same four.
They are not invented here — this model's chat template implements them, and the manager maps them
onto what it accepts:

| setting | sent to the template | what the model is told |
|---|---|---|
| `off` | `enable_thinking: false` | no reasoning step at all |
| `low` | `reasoning_effort: low` | keep thinking brief, go straight to the conclusion |
| `medium` | `reasoning_effort: medium` | nothing is injected: the model's own default |
| `high` | `reasoning_effort: xhigh` | think carefully, validate assumptions, weigh alternatives |

The template accepts **only** `low`, `medium` and `xhigh`, raising on anything else, and folds
`high` into `xhigh` itself. So four is the number of levels this model really has: a fifth would be
a second name for one of these. A single request can still override the profile with
`chat_template_kwargs`.

Optional — the desktop app: Jan v0.8.4 with the three management pages, our name and no engine
of its own:

```bash
python integrations/jan/apply.py <path to a Jan v0.8.4 checkout>
```

If the checkout was moved after `yarn install` (yarn links workspace packages with absolute
junctions), run `python tools/relink_jan_tree.py` first. Then, in that checkout, `yarn workspace @janhq/assistant-extension build`, `npx vite build` inside
`web-app/` once (it regenerates the route tree that `tsc` checks, which does not yet know the new
pages), and `yarn build:web && yarn tauri build`. If `dist/runtime` exists (section "The short
way"), apply.py stages it as a resource and the installer carries it.

## 7. Check it works

```bash
python tools/vision_probe.py                      # one image, end to end
python tools/vision_stress.py                     # 7 turns + 2 branches, the shape of three aborts
python tools/truncation_test.py --fill 70000      # 30 sections at depth; anything under 30/30 is a bug
python tools/decode_lab.py --keep --words 700,28800 --n 64
```

`vision_probe` and `vision_stress` generate their own image, so they need nothing but a running
server with a projector loaded.

## Known traps

- **`LLAMA_MMB_HC16` must be 0.** With it on, this compiler and SDK flood the output with `/`. The
  manager sets it; if you launch `llama-server` yourself, you need it too.
- **Do not measure with `llama-bench`.** It has no `--load-mode none`, so at this carve it maps
  93.7 GB through a 31.6 GB page cache and thrashes the page file. Measure through the server.
- **Do not measure decode on a server that has served an image** — it runs ~7% slower until
  restarted. See [measuring.md](measuring.md), which lists the rest.
- **GPU "Dedicated Usage" is not an allocation figure** under the flags the manager passes. Weights
  page in on demand, so it tracks how much context has been processed.
