# Jan overlay

Three management pages inside a [Jan](https://github.com/menloresearch/jan) checkout: model list,
model configuration, developer log. Optional — everything works without it; `tools/manager.py`
accepts JSON on stdin and the server is a plain OpenAI-compatible endpoint.

This is an **overlay, not a fork**. It patches a Jan checkout you install yourself, so the running
app is Jan under Jan's Apache-2.0 licence. See [NOTICE.md](../../NOTICE.md).

Base: Jan **v0.8.4**, commit `5f30aee467f08941964a83f946e2663e7ae0e01f`. `apply.py` fails rather
than guessing if an upstream anchor has moved.

## Apply

```bash
git clone --depth 1 --branch v0.8.4 https://github.com/janhq/jan src/jan
python integrations/jan/apply.py src/jan     # then build Jan as its own README describes
```

Re-running it is safe: every edit is anchored and becomes a no-op once applied.

## What it installs

| File | |
| --- | --- |
| `StrixLlamaPage.tsx` | the three pages; model catalog, launch profile, live log |
| `strixllama.css` | a bounded log viewport that pauses auto-follow when you scroll up |
| `strixllama.rs` | one Tauri command that pipes a bounded JSON request to `tools/manager.py` — no shell, no arbitrary executable |
| `locales/{en,zh-CN}/strixllama.json` | English and Chinese |
| `icons/` | the application icon, drawn by `make_icons.py` with no image library |

Language follows Jan's own setting: Jan discovers i18n namespaces with `import.meta.glob`, so the
locale files only have to be dropped in. 135 keys, identical key sets in both languages.

## What it renames

The built application is strixllama's, not Jan's: `productName`, the window title (which lives in
`tauri.windows.conf.json`, not in `index.html`), the icon, and the bundle identifier — the last of
which gives it **its own application-data directory** instead of sharing the one a real Jan install
uses. Pass `--keep-data-dir` to leave the identifier alone.

Renaming a build of Jan is what Apache-2.0 allows; [NOTICE.md](../../NOTICE.md) states plainly what
this is. Do not present it as endorsed by Jan.

`apply.py` also drops `yarn build:icon` from `build:tauri`. That step regenerates every icon size
from `icon.png` by plain downscaling, which would overwrite the 16 px icon that `make_icons.py`
deliberately draws differently — at that size the ring is under a pixel wide, so it is dropped and
the eyes grow instead.

## Building behind a GitHub block

Jan's `yarn download:bin` pulls bun and uv from GitHub releases. Where that is unreachable, fetch
the two archives through a mirror and unpack them yourself:

```bash
curl -L -o scripts/dist/bun-windows-x64.zip <mirror>/https://github.com/oven-sh/bun/releases/latest/download/bun-windows-x64.zip
curl -L -o scripts/dist/uv-x86_64-pc-windows-msvc.zip <mirror>/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
```

**Unpack them by hand as well.** `download-bin.mjs` puts the download *and* the decompress inside
one `if (!exists)`, so an archive that is already there makes it skip both, and the build then fails
on a missing `uv.exe` rather than on a missing download. The files it wants in
`src-tauri/resources/bin/` are `bun.exe`, `bun-x86_64-pc-windows-msvc.exe`, `uv.exe` and
`uv-x86_64-pc-windows-msvc.exe`. sqlite-vec failing is fine — the script marks it non-fatal.

The Jan checkout itself may also need a mirror: a `--filter=blob:none` clone cannot restore a
deleted file without reaching the origin, which is a bad surprise to discover later. Prefer
`--depth 1 --branch v0.8.4`, which is a complete tree for that one revision.

## What it removes from Jan

`apply.py` also calls `converge_settings()`, which strips settings surfaces this build has no path
through — Jan's own llama.cpp engine controls, the Hub download token, the updater, and Jan's
Resources and Community cards. They point at upstream Jan rather than at this build.

**Removing the upstream project's own links is a deliberate choice, not an oversight.** Comment out
`converge_settings()` to keep them.

## Configuration surface

The page shows what is actually worth turning: server-side thinking, context length, GPU layers, MTP
(draft model, length, threshold) and image input. Batch/UBatch/threads, concurrent slots, shared GPU
memory and n-gram drafting are behind an **Advanced** disclosure — the defaults are the measured best
for this machine, and two of those four are measured regressions kept only as escape hatches.

Flash Attention and sparse attention are stated rather than offered: turning either off runs out of
memory above 64K context on this model.

Nothing on the page reports a benchmark number. Those belong in the repository's notes, not in a
tooltip.
