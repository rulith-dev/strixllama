"""Native Jan IPC helper. Reads one JSON request on stdin; emits one JSON result.

No listening socket, shell command input, or arbitrary executable selection.
Model weights are read only. Only the pinned strixllama runtime can be managed.
"""
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'config' / 'jan'
# The one runtime: the pinned pwilkin llama.cpp branch built against the TheRock ROCm SDK in
# toolchain/rocm-venv. bootstrap/bootstrap.py produces it.
RUNTIME = ROOT / 'bin' / 'hip-rocm101' / 'llama-server.exe'
ROCM_BIN = ROOT / 'toolchain' / 'rocm-venv' / 'Lib' / 'site-packages' / '_rocm_sdk_devel' / 'bin'
# The author's launcher gates. LLAMA_MMB_HC16 must stay 0: with it on, Windows/TheRock/Clang
# floods the output with "/" (bisected in pwilkin/llama.cpp#24, reproduced here 2026-09-13).
HIP_GATES = dict(
    LLAMA_MMB=1, LLAMA_MMB_MIN_T=512, LLAMA_MMB_BF16W=1, LLAMA_MMB_GLU=1, LLAMA_MMB_TALL=2,
    LLAMA_MMB_CACHE=4, LLAMA_MMB_F32SPLIT=2, LLAMA_MMB_HC16=0, LLAMA_MMB_SHADOW=2, LLAMA_MMB_DOWN16=1,
    LLAMA_HC_CN_SHAPE=1, LLAMA_HC_GATEMIX=1, LLAMA_HC_MIX_FUSE=1, LLAMA_HC_BLK16=1,
    LLAMA_HC_RES16=1, LLAMA_HC_PACK_DI=1,
    LLAMA_NORM_GATED=1, LLAMA_NORM_ROWS=1, LLAMA_IDX_RELU_SUM=1, LLAMA_PLE_CONV=1, LLAMA_GDN_CONV=1,
    # GGML_HIP_ENABLE_UNIFIED_MEMORY is not here: it follows the profile's shared_vram switch,
    # set in runtime_environment().
    # The MTP draft context copies the target's batch sizes, so its compute buffers grow with the
    # target ubatch: at ctx 262144 / ubatch 8192 it asks for 3488 MiB and the load dies with
    # "cudaMalloc failed: out of memory". Capping the draft alone keeps both (patches/apply_spec_draft_ubatch.py).
    STRIX_SPEC_DRAFT_UBATCH=2048,
    # Sparse attention for the MTP draft head. qwen4exp.cpp gates it on n_tokens >= 128, so only the
    # draft's prefill takes it and the 1-3 query draft steps stay dense - which is what we want: the
    # draft head has no QSA of its own, so feeding a long prompt into it was quadratic, and its
    # 2048-query prefill chunks were dominated by a single dense FLASH_ATTN_EXT. It cannot change
    # what gets drafted, because the K/V a prefill stores are projections of the layer input, not of
    # the attention output. Measured on an 85K prompt: prefill 817.8 -> 856.5 t/s, decode 33.98 ->
    # 34.12 tok/s (noise), identical 203/288 draft counts, byte-identical generated text.
    # Does nothing unless the profile has MTP on.
    LLAMA_MTP_QSA=1)
# The sparse-attention gates, driven by the per-model QSA switch. LLAMA_QSA_BLOCK_SELECTION and
# LLAMA_QSA_DIRECT_INDICES together admit the block-selection path; LLAMA_QSA_SPARSE decides whether
# attention is restricted to the selected blocks or stays dense. QUERY_STRIP is a size, not a flag:
# 0 means "no strip", so it is left in place when the rest are off.
# This branch has no context threshold of its own - it engages once n_kv exceeds indexer_top_k, so
# the Jan "QSA 启用门槛" setting does not apply to the HIP runtime.
HIP_QSA_GATES = dict(
    LLAMA_QSA_SPARSE=1, LLAMA_QSA_BLOCK_SELECTION=1, LLAMA_QSA_COMPACT_METADATA=1,
    LLAMA_QSA_DENSE_SHORTCUT=1, LLAMA_QSA_DIRECT_INDICES=1, LLAMA_QSA_FA_V3=1, LLAMA_QSA_FUSE_EXPAND=1,
    LLAMA_QSA_NO_DENSE_MASK=1, LLAMA_QSA_PACK_KEYS=1, LLAMA_QSA_PACK_VALUES=1,
    LLAMA_QSA_SCORE_BOUNDS=1, LLAMA_QSA_WHOLE_ATTN=1,
    # decode-sized batches gather the selected cells instead of reading the whole cache: at 97K context
    # 49.3 vs 59.0 ms/token, and the context slope drops from 0.188 to 0.067 ms per 1000 tokens
    LLAMA_QSA_DECODE_GATHER=1,
    # Block-key cache. Safe with image input since patches/apply_qsa_kb_image_guard.py: the memory
    # reports whether it holds image cells and the graph stops wiring the cache in for that
    # conversation only, so a text-only chat keeps the cache (worth ~10% of decode).
    LLAMA_QSA_BLOCK_KEY_CACHE=1)
# The MTP draft and the attention work are specific to this model family, so the defaults that
# depend on them key off the file's name rather than one machine's path to it.
MODEL_FAMILY = 'Qwen3.8-Flash-Next'
# Unsloth's shared-Q4_K_M MTP head plus its own IQ4_XS copy of the LM head (tools/make_draft_head.py).
# The shared-* files borrow the target's Q6_K output.weight, 521 MB streamed on every draft step; the
# IQ4_XS copy is 338 MB. Every drafted token is verified by the target, so a coarser draft can only
# cost acceptance, and neither change did: Chinese 61% acceptance on the Q8_0 and the Q4_K_M base
# alike, English within its run-to-run spread (68-78% across configurations on one prompt).
# Q4_K_M over Q8_0 is worth 0.3-0.9 ms of a ~86 ms pass, i.e. under 1%; it is here for the 880 MB.
DEFAULT_DRAFT = ROOT / 'models' / 'mtp-Qwen3.8-Flash-Next-shared-Q4_K_M-head-iq4_xs.gguf'
# Vision projector shipped alongside the model (clip, projector_type qwen3vl, 904 MB F16). Without it
# llama-server has no multimodal capability at all and rejects any request carrying an image.
MMPROJ_NAME = 'mmproj-F16.gguf'
# Thinking depth -> the reasoning_effort this model's chat template accepts. 'off' is handled
# separately (enable_thinking=false). The template raises on anything outside low/medium/xhigh.
THINKING = {'off': None, 'low': 'low', 'medium': 'medium', 'high': 'xhigh'}
PORT = 8080
# the disk tier of the server's prompt cache: the default ceiling for config/jan/prompt-cache, and
# what a profile that predates the setting gets. A long conversation of this model is several GiB
# (79K tokens = 5.6 GiB, most of it context checkpoints), so this holds about three of them.
PROMPT_CACHE_DISK_MIB = 16384
# the RAM tier above it (--cache-ram). llama-server's default is 8192 MiB, which on this machine is a
# quarter of the system memory the GPU carve leaves - the KV cache itself lives in the carve, and this
# was the ~8 GB that came back when the model was unloaded. 1 GiB keeps short conversations resident;
# anything larger goes to the disk tier alone (prompt_save falls back to it when the RAM tier declines)
PROMPT_CACHE_RAM_MIB = 1024
# the disk tier writes a conversation again once it has grown by this many tokens since it was last written,
# in the background and only what changed (a few hundred MB for a 79K-token conversation, against 5.7 GB for
# the whole state): a restart loses at most this much, and a slot can be freed later without writing anything
PROMPT_CACHE_BLOCK_TOKENS = 4096
HIDDEN = 0x08000000 if os.name == 'nt' else 0
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ManagerError(ValueError):
    """An error the app can translate: `code` names it, `params` fill in the message. The text is
    English for the JSON callers; the pages render `errors.<code>` from their own locale."""
    def __init__(self, code, **params):
        self.code, self.params = code, params
        super().__init__(ERRORS[code].format(**params))


def fail(code, **params):
    raise ManagerError(code, **params)


ERRORS = {
    'not_gguf': 'Select a GGUF model file',
    'outside_roots': 'The model must be inside a registered model directory',
    'not_first_shard': 'Select the first shard of a split model',
    'gguf_truncated': 'The GGUF header is incomplete',
    'gguf_string': 'A GGUF string length is out of range',
    'gguf_nesting': 'GGUF arrays are nested too deep',
    'gguf_array': 'A GGUF array is too long',
    'gguf_type': 'Unknown GGUF field type',
    'gguf_version': 'Unsupported GGUF format',
    'gguf_count': 'The GGUF metadata count is out of range',
    'model_incomplete': 'The model is damaged or a shard is missing',
    'model_unlisted': 'The model is not in the list; rescan the model directories',
    'unknown_field': 'Unknown configuration field',
    'out_of_range': '{field} must be between {low} and {high}',
    'not_boolean': '{field} must be on or off',
    'thinking_level': 'Thinking depth must be one of {levels}',
    'draft_min': 'The MTP threshold must be between 0 and 1',
    'ubatch_gt_batch': 'ubatch cannot be larger than batch',
    'context_exceeds': 'The context is longer than the model declares',
    'kv_fixed': 'This configuration keeps the KV cache at f16',
    'flash_attention_value': 'Flash Attention must be on or off',
    'draft_path': 'The draft model path is invalid',
    'mmproj_path': 'The vision projector path is invalid',
    'mmproj_missing': 'No vision projector ({name}) beside the model: add it, pick a file, or turn image input off',
    'vision_single_slot': 'Image input needs a single slot: set parallel to 1 or turn image input off',
    'qsa_architecture': 'Sparse attention (QSA) applies to Qwen3.8 Flash Next (qwen4exp) only',
    'qsa_needs_fa': 'Sparse attention (QSA) needs Flash Attention on',
    'mtp_architecture': 'MTP is enabled for qwen4exp models only',
    'draft_missing': "No MTP draft model found: put Unsloth's mtp-*.gguf beside the model, or turn MTP off",
    'head_mismatch': 'The draft head {head} does not belong to {base}',
    'identity_changed': 'The process identity has changed; refusing to unload it',
    'log_path': 'The log path is outside the project directory',
    'port_busy': 'Port 8080 is in use by another service',
    'runtime_missing': 'The runtime is not there: {name}',
    'roots_count': 'Choose between 1 and 12 model directories',
    'root_missing': 'A model directory does not exist',
    'not_a_model': 'Drafts and vision projectors cannot be loaded as the chat model',
    'already_loaded': 'Unload the current model before loading another',
    'unknown_op': 'Unsupported operation',
    'request_too_large': 'The request is too large',
    'request_invalid': 'The request is malformed',
    'windows_only': 'This manager runs on Windows only',
    'stop_failed': 'The model process could not be stopped',
    'launch_failed': 'The model process did not start; see the log',
}
# why the last load ended, when its log said nothing more specific (status()['failure'])
FAILURES = {'oom': 'The GPU ran out of memory', 'error': 'The model process exited with an error'}
# The defaults are the measured configuration (docs/results.md), not a cautious one: context
# 262144, batch and ubatch 8192, flash attention on, and - per model, in profile() - sparse
# attention and MTP. A carve this does not fit is handled by the shared-memory fallback
# (unified_memory), so a first load succeeds either way and a fitting one is as fast as claimed.
DEFAULTS = dict(context=262144, gpu_layers=999, threads=16, batch=8192, ubatch=8192,
                # draft_max=3: swept again 2026-09-19 with the cheaper draft head, at 85K on real prose.
                # A fourth position costs 18.5 ms of an 86 ms pass (7.4 draft step + 11.2 target verify,
                # the latter being one more token's worth of expert bandwidth) and returned 0.56 tokens
                # per pass where it needed 0.64. 3 -> 27.9 ms/token, 4 -> 28.7, 5 -> 30.2.
                mtp=True, draft=str(DEFAULT_DRAFT), draft_max=3, draft_min=0.3,
                # ngram_spec off: it was measured free on prose and +27% on a coding turn, but those
                # runs had MTP off. Alongside the MTP draft it is neutral at best and costs real
                # throughput at depth - an 84K chat measured 25.50 tok/s with MTP alone against
                # 21.75 with both, the draft acceptance falling from 51% to 41% as the two drafters
                # compete for the same verification budget. On an 85K prose continuation it never
                # produced a draft at all (identical 183/229 counts with it on and off).
                # vision: load the multimodal projector, so requests may carry images. Costs 904 MB of
                # GPU memory and nothing else in this configuration - the two things llama-server turns
                # off when a projector is loaded, context shift and cache_reuse, are both already off
                # here (ctx_shift defaults false and this hybrid memory cannot shift anyway; we never
                # pass --cache-reuse). Ordinary prompt-prefix caching and MTP speculation are unaffected:
                # verified end to end, an image answered correctly with mtp on. Empty mmproj = auto.
                vision=True, mmproj='',
                # thinking: how hard the model reasons before answering, which this model's chat
                # template supports natively rather than us inventing it. 'off' sets
                # enable_thinking=false; the rest set reasoning_effort, which the template turns into
                # one injected system instruction. Its only legal values are low, medium and xhigh -
                # 'high' is an alias the template itself folds into xhigh, and 'medium' injects
                # nothing at all, i.e. the model's own default behaviour. Four levels is what this
                # model actually has; offering five would be two of them doing the same thing.
                ngram_spec=False, kv='f16', flash_attention='on', thinking='off',
                # shared_vram: force GGML_HIP_ENABLE_UNIFIED_MEMORY on. False means automatic, see
                # unified_memory(): a load first tries the dedicated carve alone, which is faster
                # and far steadier, and falls back to shared memory only when that runs out.
                qsa=False, shared_vram=False,
                # parallel: server slots. More than one costs ~12 GB of compute buffers on this model
                # (measured 13.1 GB of shared GPU memory at 4 slots against 1.1 GB at one): the worst-case
                # graph reserve uses a mixed-sequence ubatch, which fails QSA's single-sequence visibility
                # test, so the dense per-block bias and the dense KQ mask get reserved instead of the
                # compact metadata. Raise it only when concurrent requests are worth that memory; the
                # throughput is real (1/2/4 streams = 19.0/30.6/47.1 tok/s, tools/decode_concurrency.py).
                parallel=1,
                # trunk_decode_q6k: Q6_K in-memory copies of the Q8_0 trunk for decode-sized batches (HIP);
                # +2.9 GB VRAM, prefill untouched, decode -9%. Off by default so smaller carves still load.
                trunk_decode_q6k=False,
                # prompt_cache_disk: the server's prompt cache gets a disk tier (config/jan/prompt-cache,
                # PROMPT_CACHE_DISK_MIB). A finished conversation's state - ~30 KB per token for this model,
                # checkpoints included - is written when its slot is reused and read back when the
                # conversation returns, so it skips the prefill: a 34K-token session reads back in well
                # under a second against ~40 s of prefill. Survives restarts.
                prompt_cache_disk=True,
                # prompt_cache_disk_mib: the ceiling for that directory. Oldest goes first once it is
                # reached, so the only cost of a larger number is disk; 200 GiB of a 1 TB drive keeps
                # every conversation this machine can hold.
                prompt_cache_disk_mib=PROMPT_CACHE_DISK_MIB)


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


_LMSTUDIO = []


def lmstudio_models_dir():
    """LM Studio's download folder, if it is installed and the folder exists.

    This model is 93.7 GB and there is no reason to hold two copies of it. LM Studio lays models
    out as <root>/<publisher>/<repo>/<file>.gguf, which is exactly the shape catalog() walks, so
    adding its folder as a root makes everything already downloaded appear here — the same files,
    read-only, no import step and nothing copied.
    """
    if not _LMSTUDIO:
        found = None
        try:
            cfg = json.loads((Path.home() / '.lmstudio' / 'settings.json').read_text(encoding='utf-8'))
            folder = cfg.get('downloadsFolder')
            if folder and Path(folder).is_dir():
                found = str(Path(folder))
        except (OSError, ValueError):
            pass
        _LMSTUDIO.append(found)
    return _LMSTUDIO[0]


def settings():
    roots = [str(ROOT / 'models')]
    lms = lmstudio_models_dir()
    if lms:
        roots.append(lms)
    return read_json(DATA / 'settings.json', {'roots': roots, 'profiles': {}})


def identifier(path):
    return hashlib.sha256(str(Path(path).resolve()).casefold().encode()).hexdigest()[:20]


def checked_file(value):
    path = Path(value).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.gguf':
        fail('not_gguf')
    if not any(path.is_relative_to(Path(root).resolve()) for root in settings()['roots']):
        fail('outside_roots')
    if re.search(r'-(?!00001)\d{5}-of-\d{5}\.gguf$', path.name):
        fail('not_first_shard')
    return path


def mmproj_path(cfg, model):
    """The vision projector to load: the profile's own path, else the one beside THIS model.

    Beside the model actually being loaded, not beside a default one — a model kept in a second
    directory would otherwise silently load the first model's projector.
    """
    return Path(cfg['mmproj']) if cfg.get('mmproj') else Path(model['path']).parent / MMPROJ_NAME


def metadata(path):
    """Read GGUF metadata only, seeking past large tokenizer arrays."""
    with path.open('rb') as f:
        def exact(n):
            b = f.read(n)
            if len(b) != n:
                fail('gguf_truncated')
            return b
        def u32(): return struct.unpack('<I', exact(4))[0]
        def u64(): return struct.unpack('<Q', exact(8))[0]
        def string(keep=True):
            n = u64()
            if n > 16 * 1024 * 1024:
                fail('gguf_string')
            if keep:
                return exact(n).decode('utf-8', 'replace')
            f.seek(n, 1)
        fmt = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, keep=True, depth=0):
            if depth > 3: fail('gguf_nesting')
            if kind in fmt:
                code = '<' + fmt[kind]
                return struct.unpack(code, exact(struct.calcsize(code)))[0]
            if kind == 8: return string(keep)
            if kind == 9:
                item, n = u32(), u64()
                if n > 10_000_000: fail('gguf_array')
                if item in fmt:
                    f.seek(n * struct.calcsize('<' + fmt[item]), 1)
                else:
                    for _ in range(n): value(item, False, depth+1)
                return None
            fail('gguf_type')
        if exact(4) != b'GGUF' or u32() not in (2, 3):
            fail('gguf_version')
        tensors, count = u64(), u64()
        if count > 100000: fail('gguf_count')
        meta = {}
        for _ in range(count):
            key = string()
            keep = key.startswith(('general.', 'split.')) or key.endswith(('.context_length', '.block_count'))
            val = value(u32(), keep)
            if keep and val is not None: meta[key] = val
        return meta


def gguf_layout(path):
    """The byte layout of a GGUF v3 file: the key-value + tensor-info span, the alignment, the
    tensors as (name, dims, type, offset) and where the data section starts. Enough to splice two
    files together without decoding a tensor, which is all merge_draft_head needs."""
    path = Path(path)
    with path.open('rb') as f:
        def exact(n):
            b = f.read(n)
            if len(b) != n: fail('gguf_truncated')
            return b
        def u32(): return struct.unpack('<I', exact(4))[0]
        def u64(): return struct.unpack('<Q', exact(8))[0]
        fmt = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, depth=0):
            if depth > 3: fail('gguf_nesting')
            if kind in fmt:
                code = '<' + fmt[kind]
                return struct.unpack(code, exact(struct.calcsize(code)))[0]
            if kind == 8:
                n = u64()
                if n > 16 * 1024 * 1024: fail('gguf_string')
                return exact(n).decode('utf-8', 'replace')
            if kind == 9:
                item, n = u32(), u64()
                if n > 10_000_000: fail('gguf_array')
                if item in fmt: f.seek(n * struct.calcsize('<' + fmt[item]), 1)
                else:
                    for _ in range(n): value(item, depth + 1)
                return None
            fail('gguf_type')
        if exact(4) != b'GGUF' or u32() != 3: fail('gguf_version')
        n_tensors, n_kv = u64(), u64()
        if n_kv > 100000 or n_tensors > 100000: fail('gguf_count')
        kv_start, alignment = f.tell(), 32
        for _ in range(n_kv):
            key = value(8); val = value(u32())
            if key == 'general.alignment' and isinstance(val, int): alignment = val
        infos_start, tensors = f.tell(), []
        for _ in range(n_tensors):
            name = value(8); ndim = u32()
            if ndim > 8: fail('gguf_count')
            dims = [u64() for _ in range(ndim)]
            tensors.append((name, dims, u32(), u64()))
        infos_end = f.tell()
    return dict(n_kv=n_kv, span=(kv_start, infos_end), alignment=alignment, tensors=tensors,
                data_start=(infos_end + alignment - 1) // alignment * alignment, size=path.stat().st_size)


def merge_draft_head(base, head, out):
    """Write `out` = the draft `base` (Unsloth's mtp-*-shared-*.gguf) with the one tensor of `head`
    (output.weight, the draft's own quantised LM head) appended - the file tools/make_draft_head.py
    produces, made here from a downloaded 340 MB head instead of a 50 GB target shard and a
    toolchain. The base's key-value and tensor-info bytes are copied verbatim: offsets are relative
    to the data section, whose alignment is kept, so nothing has to be re-encoded."""
    base, head, out = Path(base), Path(head), Path(out)
    b, h = gguf_layout(base), gguf_layout(head)
    if (any(t[0] == 'output.weight' for t in b['tensors']) or [t[0] for t in h['tensors']] != ['output.weight']
            or metadata(head).get('general.architecture') != metadata(base).get('general.architecture')):
        fail('head_mismatch', head=head.name, base=base.name)
    name, dims, ttype, _ = h['tensors'][0]
    # a file with no tensors ends before its (aligned) data section would start, hence the clamp
    align, base_bytes = b['alignment'], max(0, b['size'] - b['data_start'])
    offset = (base_bytes + align - 1) // align * align
    info = (struct.pack('<Q', len(name.encode())) + name.encode() + struct.pack('<I', len(dims))
            + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', ttype, offset))
    part = out.with_suffix('.part')
    with base.open('rb') as src, head.open('rb') as hd, part.open('wb') as dst:
        dst.write(b'GGUF' + struct.pack('<IQQ', 3, len(b['tensors']) + 1, b['n_kv']))
        src.seek(b['span'][0]); dst.write(src.read(b['span'][1] - b['span'][0]))
        dst.write(info); dst.write(b'\0' * (-dst.tell() % align))
        src.seek(b['data_start'])
        for chunk in iter(lambda: src.read(16 << 20), b''): dst.write(chunk)
        dst.write(b'\0' * (offset - base_bytes))
        hd.seek(h['data_start'])
        for chunk in iter(lambda: hd.read(16 << 20), b''): dst.write(chunk)
    part.replace(out)
    return out


def draft_head_name(name):
    """mtp-<family>-head-<type>.gguf: a downloadable head, not a draft the loader could run."""
    low = name.lower()
    return low.startswith('mtp-') and '-head-' in low and '-shared-' not in low


def merge_heads(found):
    """For every downloaded head in the catalog whose family has Unsloth's shared draft in the same
    directory, make the merged draft once (<shared>-head-<type>.gguf) if it is not there yet.
    Returns (paths written, {head path: error})."""
    merged, errors = [], {}
    for head in found:
        if head.get('role') != 'head' or head.get('error'): continue
        low = head['filename'].lower()
        family, kind = head['filename'][4:low.index('-head-')], head['filename'][low.index('-head-') + 6:-5]
        folder = Path(head['path']).parent
        bases = [m for m in found if m.get('role') == 'draft' and not m.get('error') and Path(m['path']).parent == folder
                 and m['filename'].lower().startswith(f'mtp-{family}-shared-'.lower()) and '-head-' not in m['filename'].lower()]
        bases.sort(key=lambda m: ('Q4_K_M' not in m['filename'], m['filename']))
        if not bases: continue
        out = Path(bases[0]['path']).with_name(Path(bases[0]['path']).stem + f'-head-{kind}.gguf')
        if out.exists(): continue
        try: merged.append(str(merge_draft_head(bases[0]['path'], head['path'], out)))
        except (OSError, ValueError, struct.error) as exc: errors[head['path']] = str(exc)
    return merged, errors


def catalog(refresh=False, _after_merge=False):
    cached = read_json(DATA / 'catalog.json', None)
    if cached is not None and not refresh: return cached
    found = []
    seen = set()
    for root in settings()['roots']:
        for parent, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith('.') and not Path(parent, d).is_symlink()]
            for name in sorted(files):
                if not name.lower().endswith('.gguf') or re.search(r'-(?!00001)\d{5}-of-\d{5}\.gguf$', name): continue
                path = Path(parent, name).resolve()
                key = identifier(path)
                if key in seen: continue
                seen.add(key)
                try:
                    path = checked_file(path)
                    meta = metadata(path)
                    split = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', name)
                    shards = [path]
                    if split:
                        shards = [path.with_name(name[:split.start()] + f'-{i:05}-of-{int(split[2]):05}.gguf') for i in range(1, int(split[2])+1)]
                    missing = [p.name for p in shards if not p.exists()]
                    role = ('projection' if name.lower().startswith('mmproj') or meta.get('general.architecture') == 'clip'
                            else 'head' if draft_head_name(name) else 'draft' if name.lower().startswith('mtp-') else 'model')
                    found.append(dict(id=key, path=str(path), name=meta.get('general.name', path.stem),
                                      filename=name, architecture=meta.get('general.architecture', 'unknown'),
                                      size=sum(p.stat().st_size for p in shards if p.exists()), shards=len(shards),
                                      missing=missing, role=role, context=next((v for k,v in meta.items() if k.endswith('.context_length')), None),
                                      quant=re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I)[1]
                                      if re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I) else str(meta.get('general.file_type', ''))))
                except (OSError, ValueError, struct.error) as exc:
                    found.append(dict(id=key, path=str(path), filename=name, name=path.stem, error=str(exc), role='invalid', size=0))
    # a rescan is also when a downloaded draft head gets merged with its shared draft; the merged
    # file is then scanned like any other, once
    merged, errors = merge_heads(found) if refresh and not _after_merge else ([], {})
    if merged:
        result = catalog(True, _after_merge=True)
    else:
        result = {'models': sorted(found, key=lambda x: (x['role'], x['filename'])), 'roots': settings()['roots'], 'scanned_at': dt.datetime.now().astimezone().isoformat()}
    if refresh and not _after_merge:
        result.update(merged=merged, merge_errors=errors)
    atomic_json(DATA / 'catalog.json', result)
    return result


def model_by_id(model_id):
    for m in catalog()['models']:
        if m['id'] == model_id:
            if m.get('error') or m.get('missing'): fail('model_incomplete')
            checked_file(m['path'])
            return m
    fail('model_unlisted')


def family_draft():
    """The MTP draft for this family, if one is on disk: the repository's own first, then the best
    file under the model roots - a merged *-head-* draft (merge_draft_head) over Unsloth's
    shared-Q4_K_M over shared-Q8_0, all of which draft the same tokens. None when there is none, and
    then profile() leaves MTP off instead of pointing at a file that is not there."""
    if DEFAULT_DRAFT.is_file():
        return str(DEFAULT_DRAFT)
    drafts = [m['path'] for m in catalog()['models'] if m.get('role') == 'draft' and MODEL_FAMILY in m['filename'] and not m.get('error')]
    drafts.sort(key=lambda p: ('-head-' not in p.lower(), 'Q4_K_M' not in p, p))
    return drafts[0] if drafts else None


def profile(model):
    default = dict(DEFAULTS)
    # MTP and image input default to on, but only when their file is actually there. A load that
    # refuses to start because a companion file is missing is the wrong first experience; the
    # switch turns itself on once the file appears and the directories are rescanned.
    default['draft'] = family_draft() or ''
    default['mtp'] = MODEL_FAMILY in Path(model['path']).name and bool(default['draft'])
    default['vision'] = mmproj_path({}, model).is_file()
    # sparse attention is this architecture's, and above 64K context it is what makes the load fit
    default['qsa'] = model.get('architecture') == 'qwen4exp'
    if model.get('context'): default['context'] = min(default['context'], int(model['context']))
    saved = settings()['profiles'].get(model['id'], {})
    # A profile written by an earlier version can carry fields this one no longer has. They are
    # dropped here rather than echoed to the page, which would send them straight back and have
    # validate_profile() refuse the whole profile as unknown.
    cfg = {**default, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    # thinking was a switch before it was a level. Normalise here and not only in
    # validate_profile(): this is what the configuration page displays, and a stored `true`
    # reached it as a level called "true" whose help text does not exist.
    if type(cfg['thinking']) is bool: cfg['thinking'] = 'high' if cfg['thinking'] else 'off'
    return cfg


def validate_profile(raw, model):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS): fail('unknown_field')
    cfg = {**profile(model), **raw}
    bounds = dict(context=(512,262144), gpu_layers=(0,999), threads=(1,32), batch=(32,32768), ubatch=(32,32768), draft_max=(1,8), parallel=(1,8),
                  prompt_cache_disk_mib=(1024,262144))
    for field, (low, high) in bounds.items():
        if type(cfg[field]) is not int or not low <= cfg[field] <= high: fail('out_of_range', field=field, low=low, high=high)
    for field in ('mtp', 'ngram_spec', 'qsa', 'shared_vram', 'trunk_decode_q6k', 'vision', 'prompt_cache_disk'):
        if type(cfg[field]) is not bool: fail('not_boolean', field=field)
    # thinking was a switch before it was a level; a profile saved back then still loads
    if type(cfg['thinking']) is bool: cfg['thinking'] = 'high' if cfg['thinking'] else 'off'
    if cfg['thinking'] not in THINKING: fail('thinking_level', levels=', '.join(THINKING))
    if type(cfg['draft_min']) not in (int,float) or not 0 <= cfg['draft_min'] <= 1: fail('draft_min')
    if cfg['ubatch'] > cfg['batch']: fail('ubatch_gt_batch')
    if model.get('context') and cfg['context'] > model['context']: fail('context_exceeds')
    if cfg['kv'] != 'f16': fail('kv_fixed')
    if cfg['flash_attention'] not in ('on', 'off'): fail('flash_attention_value')
    if not isinstance(cfg['draft'], str): fail('draft_path')
    if not isinstance(cfg['mmproj'], str): fail('mmproj_path')
    if cfg['vision']:
        # An image's cells repeat one position (M-RoPE) and run ahead of their cells, so the sparse
        # attention's block enumeration has to rank them, and it can only do that while the cache holds
        # one sequence: with two conversations resident an image aborts the server in set_input_qsa
        # (`oor`). Several slots therefore exclude image input until that path exists.
        if cfg.get('parallel', 1) > 1: fail('vision_single_slot')
        # a path the user picked goes through the full check (it must sit in a registered model root);
        # the automatic one ships beside the model, so only its existence matters
        if cfg['mmproj']: checked_file(cfg['mmproj'])
        elif not mmproj_path(cfg, model).is_file(): fail('mmproj_missing', name=MMPROJ_NAME)
    if cfg['qsa']:
        if model.get('architecture') != 'qwen4exp':
            fail('qsa_architecture')
        if cfg['flash_attention'] != 'on': fail('qsa_needs_fa')
    if cfg['mtp']:
        if model['architecture'] != 'qwen4exp': fail('mtp_architecture')
        if not cfg['draft'] or not Path(cfg['draft']).is_file(): fail('draft_missing')
        checked_file(cfg['draft'])
    return cfg


def bundled_rocm():
    """The installed layout: the ROCm DLLs sit beside llama-server (tools/make_runtime_bundle.py),
    so Windows finds them without a PATH entry and there is no SDK directory at all."""
    return (RUNTIME.parent / 'amdhip64_7.dll').is_file()


def runtime_available():
    return (RUNTIME.is_file() and (RUNTIME.parent/'ggml-hip.dll').is_file()
            and (bundled_rocm() or (ROCM_BIN.is_dir() and (ROCM_BIN/'amdhip64_7.dll').is_file())))


def selected_runtime(cfg):
    return RUNTIME


def managed_runtime(path):
    return bool(path) and Path(path).resolve() == RUNTIME.resolve()


def dedicated_vram_bytes():
    """The GPU's dedicated memory as the display driver registered it - on this machine, the BIOS
    carve. Read from the registry rather than asked of HIP, so it costs nothing and needs no GPU
    context. None when it cannot be read (not Windows, no adapter entry)."""
    try:
        import winreg
    except ImportError:
        return None
    best = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}') as adapters:
            for i in range(64):
                try: name = winreg.EnumKey(adapters, i)
                except OSError: break
                try:
                    with winreg.OpenKey(adapters, name) as adapter:
                        size = winreg.QueryValueEx(adapter, 'HardwareInformation.qwMemorySize')[0]
                except OSError: continue
                if isinstance(size, int) and size > (best or 0): best = size
    except OSError:
        return None
    return best


def memory_fingerprint(cfg):
    """The profile fields that decide how much device memory a load takes."""
    return {k: cfg.get(k) for k in ('context', 'batch', 'ubatch', 'parallel', 'gpu_layers', 'mtp', 'draft', 'vision', 'mmproj', 'trunk_decode_q6k')}


def unified_memory(model, cfg):
    """Whether this load runs with GGML_HIP_ENABLE_UNIFIED_MEMORY.

    Off keeps every allocation in the dedicated carve and is the first attempt whenever the profile
    does not force it on: it is the faster setting, and the steadier one by a wider margin (see
    runtime_environment). When a load has already died of out-of-memory with this model, at this
    carve, with these memory-relevant settings, that is remembered (settings.json, shared_vram_auto)
    and the next load starts in shared memory instead of failing the same way again. Change the
    carve or any of those settings and it is tried afresh, so a bigger carve gets its speed back.
    """
    if cfg.get('shared_vram', False):
        return True
    remembered = settings().get('shared_vram_auto', {}).get(model['id'])
    return bool(remembered) and remembered.get('dedicated') == dedicated_vram_bytes() \
        and remembered.get('fingerprint') == memory_fingerprint(cfg)


def remember_shared_vram(model, cfg):
    all_cfg = settings()
    all_cfg.setdefault('shared_vram_auto', {})[model['id']] = dict(
        dedicated=dedicated_vram_bytes(), fingerprint=memory_fingerprint(cfg),
        since=dt.datetime.now().astimezone().isoformat())
    atomic_json(DATA / 'settings.json', all_cfg)


# How the runtime says it ran out of device memory: ggml's allocator ("cudaMalloc failed: out of
# memory"), the KV cache and graph reserves ("failed to allocate ... buffer"), and HIP's own name.
OOM_SIGNS = ('out of memory', 'cudamalloc failed', 'hiperroroutofmemory', 'failed to allocate')


def exit_reason(log):
    """Why a load ended, from the tail of its log: ('oom', line) when it ran out of memory, ('error',
    line) for the last line that looks like one, ('exited', '') when the log says nothing."""
    try:
        path = Path(log)
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 65536))
            tail = f.read().decode('utf-8', 'replace')
    except OSError:
        return 'exited', ''
    lines = [l.strip() for l in reversed(tail.splitlines()) if l.strip()]
    oom = next((l for l in lines if any(s in l.lower() for s in OOM_SIGNS)), None)
    if oom: return 'oom', oom
    err = next((l for l in lines if re.search(r'\berror\b|failed|abort|exception|\bE\b', l, re.I)), None)
    return ('error', err) if err else ('exited', '')


def runtime_environment(cfg, unified=False):
    # start from a clean slate: a stray LLAMA_*/GGML_*/STRIX_* from a shell would silently change
    # the graph, and an inherited value is never what the profile asked for
    env = {k: v for k, v in os.environ.copy().items()
           if not k.upper().startswith(('LLAMA_', 'GGML_', 'STRIX_'))}
    env.update({k: str(v) for k, v in HIP_GATES.items()})
    on = cfg.get('qsa', False)
    env.update({k: ('1' if on else '0') for k in HIP_QSA_GATES})
    env['LLAMA_QSA_QUERY_STRIP'] = '512' if on else '0'
    # Off keeps allocations in the dedicated carve. It does not eliminate shared-memory use
    # entirely - Task Manager still shows a few GB - but it stops the spill that costs prefill:
    # pp16384 903.64 +/- 3.71 off against 867.12 +/- 24.11 on, three llama-bench reps. Whether a
    # load gets it is decided by unified_memory(), not read from the profile here.
    env['GGML_HIP_ENABLE_UNIFIED_MEMORY'] = '1' if unified else '0'
    # Q6_K decode twins of the Q8_0 trunk (llama-model.cpp build_decode_twins): batches of <= 8 tokens
    # read 23% fewer trunk bytes; prefill keeps the Q8_0 originals. No UI control - measured
    # prefill-neutral and 4% on decode for 2.9 GB, and at ctx 262144 it can stop a long prompt loading.
    env['LLAMA_TRUNK_DECODE_Q6K'] = '1' if cfg.get('trunk_decode_q6k', False) else '0'
    # the server's prompt cache gets a disk tier (see DEFAULTS); the directory lives with the settings
    if cfg.get('prompt_cache_disk', False):
        env['STRIX_PROMPT_CACHE_DIR'] = str(DATA / 'prompt-cache')
        env['STRIX_PROMPT_CACHE_MIB'] = str(cfg.get('prompt_cache_disk_mib') or PROMPT_CACHE_DISK_MIB)
        env['STRIX_PROMPT_CACHE_BLOCK'] = str(PROMPT_CACHE_BLOCK_TOKENS)
    if not bundled_rocm():
        env['PATH'] = str(ROCM_BIN) + os.pathsep + os.environ.get('PATH', '')
    return env


def argv(model, cfg):
    args = [str(selected_runtime(cfg)), '-m', model['path'], '-ngl', str(cfg['gpu_layers']), '-c', str(cfg['context']),
            '-b', str(cfg['batch']), '-ub', str(cfg['ubatch']), '-t', str(cfg['threads']), '--poll', '0',
            '--fit', 'off', '-np', str(cfg.get('parallel', 1)), '-fa', cfg['flash_attention'], '-ctk', 'f16', '-ctv', 'f16', '--jinja',
            '--host', '127.0.0.1', '--port', str(PORT)]
    if cfg.get('parallel', 1) > 1:
        # without it the pool is split evenly and each slot would see context/parallel tokens; -kvu keeps one
        # shared pool so a single conversation can still use the whole context when the others are idle
        args += ['-kvu']
    think = cfg['thinking'] if not isinstance(cfg['thinking'], bool) else ('high' if cfg['thinking'] else 'off')
    kwargs = ({'enable_thinking': False} if think == 'off'
              else {'enable_thinking': True, 'reasoning_effort': THINKING[think]})
    # --no-cache-idle-slots: upstream saves AND clears every idle slot on each new task when the KV is unified
    # (-kvu, i.e. more than one slot), so talking to A, then B, then A read A back from disk and wrote B out,
    # ~6 s a switch for long conversations, although the cells were already allocated. Without it they stay
    # in their slots until the pool is actually full; the disk tier writes them in the background instead.
    args += ['--cache-prompt', '--cache-ram', str(PROMPT_CACHE_RAM_MIB), '--no-cache-idle-slots',
             '--chat-template-kwargs', json.dumps(kwargs, separators=(',',':'))]
    if cfg.get('vision', False):
        args += ['--mmproj', str(mmproj_path(cfg, model))]
    # this runtime reads the per-layer embedding table itself with offset I/O, so it must not be
    # pinned to CPU memory
    args += ['--load-mode', 'none', '--lazy-mode', 'on-direct']
    spec_types = []
    if cfg.get('ngram_spec', False):
        spec_types.append('ngram-mod')
        # Bound recurrent rollback storage separately from the MTP draft length.
        args += ['--spec-ngram-mod-n-match', '24', '--spec-ngram-mod-n-min', '4', '--spec-ngram-mod-n-max', '8']
    if cfg['mtp']:
        spec_types.append('draft-mtp')
        args += ['-md', cfg['draft'], '-ngld', str(cfg['gpu_layers']), '--spec-draft-n-max', str(cfg['draft_max']), '--spec-draft-p-min', str(cfg['draft_min'])]
    if spec_types: args += ['--spec-type', ','.join(spec_types)]
    return args


def process_identity(pid, terminate=False, expected=None):
    """Hold the same process handle while checking identity and terminating it."""
    if os.name != 'nt': fail('windows_only')
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    k.TerminateProcess.argtypes = [wintypes.HANDLE,wintypes.UINT]
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    handle = k.OpenProcess(0x1000 | 0x100000 | (1 if terminate else 0), False, int(pid))
    if not handle: return None
    try:
        buf, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not k.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)): return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]): return None
        birth = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
        result = {'pid':int(pid), 'exe':str(Path(buf.value).resolve()), 'birth':birth}
        if terminate:
            # the identity (pid, executable, birth time) is what was adopted or started; that it
            # matches, and that it is a llama-server at all, is the check - so a server another
            # copy of this manager started can be unloaded too, and nothing else ever is
            if result != expected or Path(result['exe']).name.lower() != 'llama-server.exe':
                fail('identity_changed')
            if not k.TerminateProcess(handle,0): fail('stop_failed')
            k.WaitForSingleObject(handle,10000)
        return result
    finally: k.CloseHandle(handle)


def discover():
    script = "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process -Filter \"Name = 'llama-server.exe'\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    p = subprocess.run(['powershell.exe','-NoProfile','-Command',script],capture_output=True,encoding='utf-8',errors='replace',creationflags=HIDDEN,timeout=15)
    entries = json.loads(p.stdout) if p.stdout.strip() else []
    if isinstance(entries,dict): entries=[entries]
    return entries


def state():
    saved = read_json(DATA / 'process.json', {})
    ident = saved.get('identity')
    if ident and managed_runtime(ident.get('exe')) and process_identity(ident['pid']) == ident: return saved
    if ident and saved.get('adopted') is False and saved.get('log'):
        # A process this manager started is gone without a stop. Record why, once: status() acts
        # on an out-of-memory (shared-memory fallback) and the page can say what happened instead
        # of silently going back to "not loaded".
        reason, line = exit_reason(saved['log'])
        saved = {'last_log': saved['log'], 'exited': dict(reason=reason, line=line, model_id=saved.get('model_id'),
                 profile=saved.get('profile'), unified=bool(saved.get('unified')), log=saved['log'])}
        atomic_json(DATA / 'process.json', saved)
        return saved
    # Adopt a server of ours on the configured local endpoint: the exact project binary, or a
    # llama-server on our port started with our launch flags by another copy of this manager (an
    # earlier install, a checkout in another directory). Without the second case a model loaded
    # from one copy could not be unloaded from the next, and its port stayed taken.
    for p in discover():
        cmd = p.get('CommandLine') or ''
        ours = managed_runtime(p.get('ExecutablePath')) or (
            Path(p.get('ExecutablePath') or '').name.lower() == 'llama-server.exe' and '--lazy-mode on-direct' in cmd)
        if ours and re.search(r'--port\s+8080(?:\s|$)', cmd) and re.search(r'--host\s+127\.0\.0\.1(?:\s|$)', cmd):
            model_match = re.search(r'(?:^|\s)-m\s+(?:"([^"]+)"|(\S+))', cmd)
            log_match = re.search(r'--log-file\s+(?:"([^"]+)"|(\S+))', cmd)
            identity = process_identity(p['ProcessId'])
            if identity and Path(identity['exe']) == Path(p['ExecutablePath']).resolve():
                saved = dict(identity=identity, model_path=next((v for v in model_match.groups() if v), '') if model_match else '',
                             log=next((v for v in log_match.groups() if v), '') if log_match else '', command=cmd, adopted=True)
                atomic_json(DATA/'process.json',saved)
                return saved
    result = {'last_log':saved.get('log', saved.get('last_log',''))}
    if saved.get('exited'): result['exited'] = saved['exited']
    return result


def http_json(path):
    with HTTP.open(f'http://127.0.0.1:{PORT}'+path,timeout=1.5) as res: return json.load(res)


def status():
    s = state()
    exited = s.get('exited')
    if exited and exited.get('reason') == 'oom' and not exited.get('unified') and exited.get('model_id') and exited.get('profile') is not None:
        # The load ran out of the dedicated carve. Remember that for this model, carve and profile,
        # and load again in shared memory - once: the relaunch records unified=True, so if that
        # dies too the failure is reported rather than retried.
        try:
            m = model_by_id(exited['model_id']); cfg = validate_profile(exited['profile'], m)
            remember_shared_vram(m, cfg)
            s = launch(m, cfg, unified=True, notice='shared_vram_fallback')
        except Exception as exc:
            s = {**s, 'exited': {**exited, 'reason': 'error', 'line': f'{exited.get("line", "")}; reloading in shared memory failed: {exc}'}}
            atomic_json(DATA / 'process.json', s)
    result = {**s, 'status':'stopped', 'endpoint':f'http://127.0.0.1:{PORT}/v1',
              'runtime':s.get('identity', {}).get('exe', str(RUNTIME)),
              'runtime_available': runtime_available(), 'dedicated_vram': dedicated_vram_bytes()}
    e = s.get('exited') or {}
    if e.get('reason') in ('oom', 'error'):
        # a plain exit with nothing in the log is not reported: the app closing takes the server
        # with it, and "the last load failed" would be the wrong thing to say about that
        result['failure'] = e.get('line') or FAILURES[e['reason']]
        # the code only when the text is ours to translate, not a line quoted from the log
        result['failure_code'] = None if e.get('line') else e['reason']
    if s.get('identity'):
        result['status']='loading'
        result['model_name'] = next((x.get('name') for x in catalog()['models'] if x.get('path') == s.get('model_path')), None) or Path(s.get('model_path', '')).stem
        try:
            if http_json('/health').get('status')=='ok':
                result['status']='ready'
                models=http_json('/v1/models')['data']
                result['served_models']=models
        except Exception: pass
    return result


def logs(offset=0):
    s = state()
    path = Path(s.get('log') or s.get('last_log') or ROOT/'logs'/'not-started.log')
    if not path.resolve().is_relative_to((ROOT/'logs').resolve()): fail('log_path')
    if not path.exists(): return {'text':'','offset':0,'file':str(path),'reset':False}
    size = path.stat().st_size
    offset = max(0,int(offset))
    reset = offset > size
    if reset: offset=0
    if offset == 0: offset=max(0,size-65536)
    with path.open('rb') as f:
        f.seek(offset)
        b=f.read(65536)
        # Do not split a UTF-8 sequence across polling boundaries.
        tail=0
        for n in range(0,4):
            try: text=b[:len(b)-n if n else None].decode('utf-8'); tail=n; break
            except UnicodeDecodeError:
                if n==3: text=b.decode('utf-8','replace')
        return {'text':text,'offset':offset+len(b)-tail,'file':str(path),'reset':reset}


def launch(m, cfg, unified, notice=None):
    """Start the runtime for model m with the already validated profile cfg: check the port, write
    the log banner, record the process identity. `unified` is the shared-memory decision."""
    try: http_json('/health'); fail('port_busy')
    except (urllib.error.URLError,TimeoutError): pass
    # Check port before allocating model memory; do not stop unrelated engines.
    import socket
    with socket.socket() as sock:
        try: sock.bind(('127.0.0.1',PORT))
        except OSError: fail('port_busy')
    runtime = selected_runtime(cfg)
    if not runtime.is_file() or not (runtime.parent/'ggml-hip.dll').is_file():
        fail('runtime_missing', name=runtime.parent.name)
    log=ROOT/'logs'/('jan-managed-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.log')
    log.parent.mkdir(exist_ok=True)
    command=argv(m,cfg)
    env=runtime_environment(cfg, unified)
    banner = (f'[strixllama] runtime={runtime.parent.name} (HIP/ROCm); LLAMA_MMB_HC16={env["LLAMA_MMB_HC16"]} '
              f'(must stay 0 on Windows); gates={sum(1 for k in env if k.startswith("LLAMA_"))}; '
              f'QSA={"on" if cfg["qsa"] else "off"} (this runtime has no context threshold); '
              f'MTP={"on" if cfg["mtp"] else "off"}'
              f'{f" (draft ubatch capped to {env['STRIX_SPEC_DRAFT_UBATCH']})" if cfg["mtp"] else ""}; '
              f'n-gram draft={"on (match=24, min=4, max=8)" if cfg["ngram_spec"] else "off"}; '
              f'vision={"on" if cfg["vision"] else "off"}; shared memory={"on" if unified else "off"}; '
              f'PLE reader=on-direct; rocm={"bundled beside the server" if bundled_rocm() else ROCM_BIN}\n')
    with log.open('wb') as f:
        f.write(banner.encode('utf-8'))
        f.flush()
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=HIDDEN,cwd=ROOT)
    ident=process_identity(proc.pid)
    if not ident: fail('launch_failed')
    saved=dict(identity=ident,model_id=m['id'],model_path=m['path'],log=str(log),command=subprocess.list2cmdline(command),profile=cfg,
               unified=unified,started_at=dt.datetime.now().astimezone().isoformat(),adopted=False,
               runtime_env={k:v for k,v in env.items() if k.startswith(('LLAMA_','GGML_','STRIX_'))})
    if notice: saved['notice']=notice
    atomic_json(DATA/'process.json',saved)
    return saved


def handle(op, data):
    if op=='catalog': return catalog(bool(data.get('refresh')))
    if op=='status': return status()
    if op=='logs': return logs(data.get('offset',0))
    if op=='profile':
        m=model_by_id(data['id'])
        # what the companion switches can be turned on with: the page explains an off switch by it
        companions={'draft':family_draft(),'mmproj':mmproj_path({},m).is_file(),
                    'draft_head':any('-head-' in Path(p).name.lower() for p in [family_draft() or ''])}
        return {'model':m,'profile':profile(m),'companions':companions}
    if op=='roots':
        roots=data['roots']
        if not isinstance(roots,list) or not 1 <= len(roots) <= 12: fail('roots_count')
        parsed=[str(Path(r).resolve(strict=True)) for r in roots]
        if any(not Path(r).is_dir() for r in parsed): fail('root_missing')
        cfg=settings();cfg['roots']=list(dict.fromkeys(parsed));atomic_json(DATA/'settings.json',cfg)
        return catalog(True)
    if op=='save':
        m=model_by_id(data['id']); cfg=validate_profile(data['profile'],m)
        all_cfg=settings();all_cfg['profiles'][m['id']]=cfg;atomic_json(DATA/'settings.json',all_cfg)
        return {'profile':cfg,'restart_required':bool(state().get('identity')),'argv':argv(m,cfg)}
    if op=='stop':
        s=state()
        if s.get('identity'): process_identity(s['identity']['pid'],True,s['identity'])
        atomic_json(DATA/'process.json',{'last_log':s.get('log',s.get('last_log',''))})
        return {'status':'stopped'}
    if op=='start':
        m=model_by_id(data['id'])
        if m['role']!='model': fail('not_a_model')
        cfg=validate_profile(data.get('profile',profile(m)),m)
        if state().get('identity'): fail('already_loaded')
        return {**launch(m,cfg,unified_memory(m,cfg)),'status':'loading'}
    fail('unknown_op')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536: fail('request_too_large')
        request=json.loads(raw)
        if not isinstance(request,dict): fail('request_invalid')
        # Serialize all access across the short-lived native IPC helpers.
        DATA.mkdir(parents=True,exist_ok=True)
        with (DATA/'manager.lock').open('a+b') as lock:
            import msvcrt
            if lock.tell()==0: lock.write(b'0');lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
            try: result=handle(request['op'],request.get('data',{}))
            finally: lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
        print(json.dumps({'ok':True,'data':result},ensure_ascii=False))
    except ManagerError as exc:
        print(json.dumps({'ok':False,'error':str(exc),'code':exc.code,'params':exc.params},ensure_ascii=False))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False))
        sys.exit(1)


if __name__=='__main__': main()
