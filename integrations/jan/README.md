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
python integrations/jan/apply.py     # then build Jan as its own README describes
```

## What it installs

| File | |
| --- | --- |
| `StrixLlamaPage.tsx` | the three pages; model catalog, launch profile, live log |
| `strixllama.css` | a bounded log viewport that pauses auto-follow when you scroll up |
| `strixllama.rs` | one Tauri command that pipes a bounded JSON request to `tools/manager.py` — no shell, no arbitrary executable |
| `locales/{en,zh-CN}/strixllama.json` | English and Chinese |

Language follows Jan's own setting: Jan discovers i18n namespaces with `import.meta.glob`, so the
locale files only have to be dropped in. 115 keys, identical key sets in both languages.

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
