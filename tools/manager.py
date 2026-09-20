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
HIDDEN = 0x08000000 if os.name == 'nt' else 0
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
                trunk_decode_q6k=False)


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
        raise ValueError('请选择 GGUF 模型文件')
    if not any(path.is_relative_to(Path(root).resolve()) for root in settings()['roots']):
        raise ValueError('模型必须位于已登记的模型目录中')
    if re.search(r'-(?!00001)\d{5}-of-\d{5}\.gguf$', path.name):
        raise ValueError('分片模型必须选择第一个分片')
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
                raise ValueError('GGUF 文件头不完整')
            return b
        def u32(): return struct.unpack('<I', exact(4))[0]
        def u64(): return struct.unpack('<Q', exact(8))[0]
        def string(keep=True):
            n = u64()
            if n > 16 * 1024 * 1024:
                raise ValueError('GGUF 字符串长度异常')
            if keep:
                return exact(n).decode('utf-8', 'replace')
            f.seek(n, 1)
        fmt = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def value(kind, keep=True, depth=0):
            if depth > 3: raise ValueError('GGUF 数组嵌套异常')
            if kind in fmt:
                code = '<' + fmt[kind]
                return struct.unpack(code, exact(struct.calcsize(code)))[0]
            if kind == 8: return string(keep)
            if kind == 9:
                item, n = u32(), u64()
                if n > 10_000_000: raise ValueError('GGUF 数组过长')
                if item in fmt:
                    f.seek(n * struct.calcsize('<' + fmt[item]), 1)
                else:
                    for _ in range(n): value(item, False, depth+1)
                return None
            raise ValueError('未知 GGUF 字段类型')
        if exact(4) != b'GGUF' or u32() not in (2, 3):
            raise ValueError('不支持的 GGUF 格式')
        tensors, count = u64(), u64()
        if count > 100000: raise ValueError('GGUF 元数据数量异常')
        meta = {}
        for _ in range(count):
            key = string()
            keep = key.startswith(('general.', 'split.')) or key.endswith(('.context_length', '.block_count'))
            val = value(u32(), keep)
            if keep and val is not None: meta[key] = val
        return meta


def catalog(refresh=False):
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
                    role = 'projection' if name.lower().startswith('mmproj') or meta.get('general.architecture') == 'clip' else 'draft' if name.lower().startswith('mtp-') else 'model'
                    found.append(dict(id=key, path=str(path), name=meta.get('general.name', path.stem),
                                      filename=name, architecture=meta.get('general.architecture', 'unknown'),
                                      size=sum(p.stat().st_size for p in shards if p.exists()), shards=len(shards),
                                      missing=missing, role=role, context=next((v for k,v in meta.items() if k.endswith('.context_length')), None),
                                      quant=re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I)[1]
                                      if re.search(r'(?:UD-)?((?:IQ|Q|MXFP|BF|F)[A-Z0-9_]+)(?:-\d{5}-of|\.gguf)', name, re.I) else str(meta.get('general.file_type', ''))))
                except (OSError, ValueError, struct.error) as exc:
                    found.append(dict(id=key, path=str(path), filename=name, name=path.stem, error=str(exc), role='invalid', size=0))
    result = {'models': sorted(found, key=lambda x: (x['role'], x['filename'])), 'roots': settings()['roots'], 'scanned_at': dt.datetime.now().astimezone().isoformat()}
    atomic_json(DATA / 'catalog.json', result)
    return result


def model_by_id(model_id):
    for m in catalog()['models']:
        if m['id'] == model_id:
            if m.get('error') or m.get('missing'): raise ValueError('模型损坏或缺少分片')
            checked_file(m['path'])
            return m
    raise ValueError('模型不在列表中，请重新扫描')


def family_draft():
    """The draft head to use when the repository's own is not there: on an installed copy there is
    no models/ directory of ours, but Unsloth's shared-Q4_K_M MTP file ships beside the model, and it
    drafts the same tokens (see DEFAULT_DRAFT) at under 1% of the pass. Ours is preferred when both
    are present."""
    if DEFAULT_DRAFT.is_file():
        return str(DEFAULT_DRAFT)
    drafts = [m['path'] for m in catalog()['models'] if m.get('role') == 'draft' and MODEL_FAMILY in m['filename'] and not m.get('error')]
    drafts.sort(key=lambda p: ('head-iq4_xs' not in p, 'Q4_K_M' not in p, p))
    return drafts[0] if drafts else str(DEFAULT_DRAFT)


def profile(model):
    default = dict(DEFAULTS)
    default['mtp'] = MODEL_FAMILY in Path(model['path']).name
    default['draft'] = family_draft()
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
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS): raise ValueError('未知配置字段')
    cfg = {**profile(model), **raw}
    bounds = dict(context=(512,262144), gpu_layers=(0,999), threads=(1,32), batch=(32,32768), ubatch=(32,32768), draft_max=(1,8), parallel=(1,8))
    for field, (low, high) in bounds.items():
        if type(cfg[field]) is not int or not low <= cfg[field] <= high: raise ValueError(f'{field} 应在 {low}–{high} 之间')
    for field in ('mtp', 'ngram_spec', 'qsa', 'shared_vram', 'trunk_decode_q6k', 'vision'):
        if type(cfg[field]) is not bool: raise ValueError(f'{field} 必须为开关值')
    # thinking was a switch before it was a level; a profile saved back then still loads
    if type(cfg['thinking']) is bool: cfg['thinking'] = 'high' if cfg['thinking'] else 'off'
    if cfg['thinking'] not in THINKING: raise ValueError('思考深度应为 ' + '、'.join(THINKING))
    if type(cfg['draft_min']) not in (int,float) or not 0 <= cfg['draft_min'] <= 1: raise ValueError('MTP 阈值应在 0–1 之间')
    if cfg['ubatch'] > cfg['batch']: raise ValueError('ubatch 不能大于 batch')
    if model.get('context') and cfg['context'] > model['context']: raise ValueError('上下文超过模型声明长度')
    if cfg['kv'] != 'f16': raise ValueError('当前配置固定 f16 KV')
    if cfg['flash_attention'] not in ('on', 'off'): raise ValueError('Flash Attention 应为 on 或 off')
    if not isinstance(cfg['draft'], str): raise ValueError('草稿模型路径无效')
    if not isinstance(cfg['mmproj'], str): raise ValueError('视觉投影模型路径无效')
    if cfg['vision']:
        # a path the user picked goes through the full check (it must sit in a registered model root);
        # the automatic one ships beside the model, so only its existence matters
        if cfg['mmproj']: checked_file(cfg['mmproj'])
        elif not mmproj_path(cfg, model).is_file(): raise ValueError('未找到随模型附带的视觉投影模型，请指定文件或关闭图像输入')
    if cfg['qsa']:
        if model.get('architecture') != 'qwen4exp':
            raise ValueError('QSA 优化仅适用于 Qwen3.8 Flash Next（qwen4exp）')
        if cfg['flash_attention'] != 'on': raise ValueError('QSA 优化需要开启 Flash Attention')
    if cfg['mtp']:
        if model['architecture'] != 'qwen4exp': raise ValueError('此第一版只对 qwen4exp 启用 MTP')
        try: checked_file(cfg['draft'])
        except (FileNotFoundError, OSError): raise ValueError('未找到 MTP 草稿模型：把 Unsloth 的 mtp-*.gguf 放到模型旁边，或关闭 MTP')
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
    args += ['--cache-prompt', '--chat-template-kwargs', json.dumps(kwargs, separators=(',',':'))]
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
    if os.name != 'nt': raise RuntimeError('当前管理器仅支持 Windows')
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
                raise ValueError('进程身份已变化，拒绝卸载')
            if not k.TerminateProcess(handle,0): raise OSError('无法停止模型进程')
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
            s = {**s, 'exited': {**exited, 'reason': 'error', 'line': f'{exited.get("line", "")}；改用共享显存重新加载失败：{exc}'}}
            atomic_json(DATA / 'process.json', s)
    result = {**s, 'status':'stopped', 'endpoint':f'http://127.0.0.1:{PORT}/v1',
              'runtime':s.get('identity', {}).get('exe', str(RUNTIME)),
              'runtime_available': runtime_available(), 'dedicated_vram': dedicated_vram_bytes()}
    e = s.get('exited') or {}
    if e.get('reason') in ('oom', 'error'):
        # a plain exit with nothing in the log is not reported: the app closing takes the server
        # with it, and "the last load failed" would be the wrong thing to say about that
        result['failure'] = e.get('line') or {'oom': '显存不足', 'error': '模型进程报错退出'}[e['reason']]
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
    if not path.resolve().is_relative_to((ROOT/'logs').resolve()): raise ValueError('日志路径超出项目目录')
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
    try: http_json('/health'); raise ValueError('8080 端口正被其他服务占用')
    except (urllib.error.URLError,TimeoutError): pass
    # Check port before allocating model memory; do not stop unrelated engines.
    import socket
    with socket.socket() as sock:
        try: sock.bind(('127.0.0.1',PORT))
        except OSError: raise ValueError('8080 端口正被其他服务占用')
    runtime = selected_runtime(cfg)
    if not runtime.is_file() or not (runtime.parent/'ggml-hip.dll').is_file():
        raise ValueError(f'优化运行时不存在：{runtime.parent.name}')
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
    if not ident: raise RuntimeError('模型进程启动失败，请查看日志')
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
        m=model_by_id(data['id']); return {'model':m,'profile':profile(m)}
    if op=='roots':
        roots=data['roots']
        if not isinstance(roots,list) or not 1 <= len(roots) <= 12: raise ValueError('请选择 1–12 个模型目录')
        parsed=[str(Path(r).resolve(strict=True)) for r in roots]
        if any(not Path(r).is_dir() for r in parsed): raise ValueError('模型目录不存在')
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
        if m['role']!='model': raise ValueError('草稿和视觉投影不能单独作为聊天模型加载')
        cfg=validate_profile(data.get('profile',profile(m)),m)
        if state().get('identity'): raise ValueError('请先卸载当前模型，再加载所选模型')
        return {**launch(m,cfg,unified_memory(m,cfg)),'status':'loading'}
    raise ValueError('不支持的管理操作')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536: raise ValueError('请求过大')
        request=json.loads(raw)
        if not isinstance(request,dict): raise ValueError('请求格式无效')
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
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False))
        sys.exit(1)


if __name__=='__main__': main()
