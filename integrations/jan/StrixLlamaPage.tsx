import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from '@tanstack/react-router'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { request, describeError, useStrixLlamaStatus, type Profile } from './status'
import { Database, SlidersHorizontal, Terminal, RefreshCw, Play, Square, Search, Copy, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import './strixllama.css'

type Model = { id: string; path: string; name: string; filename: string; architecture?: string; quant?: string; size: number; shards?: number; missing?: string[]; role: string; error?: string; context?: number }
// trunk_decode_q6k stays in the type and in the manager: existing profiles keep working and the
// switch is still reachable from the config file. It has no control here - measured prefill-neutral
// and 4% on decode for 2.9 GB, and at the production context it makes long prompts fail to load.

// The four levels this model's chat template actually has: it accepts low, medium and xhigh, folds
// 'high' into xhigh, and injects nothing at all for medium. A fifth level would be a duplicate.
const THINKING_LEVELS = ['off', 'low', 'medium', 'high']
// merged / merge_errors: set by a rescan, when a downloaded draft head was combined with its shared draft
type Catalog = { models: Model[]; roots: string[]; scanned_at: string; merged?: string[]; merge_errors?: Record<string, string> }
// what the companion switches have to work with: the draft the profile would use (null when none is
// on disk), whether it is a merged *-head-* one, and whether the projector sits beside the model
type Companions = { draft: string | null; draft_head: boolean; mmproj: boolean }
type LogChunk = { text: string; offset: number; file: string; reset: boolean }
type View = 'models' | 'configuration' | 'developer'
const gb = (n: number) => `${(n / 1e9).toFixed(1)} GB`
// Shared GPU memory has no control and no readout here: the manager decides it per load (see
// unified_memory() in tools/manager.py) and says so in a notice only when it had to fall back.

export default function StrixLlamaPage({ view }: { view: View }) {
  const { t } = useTranslation()
  const tr = useCallback((key: string, vars?: Record<string, unknown>) => t(`strixllama:${key}`, vars), [t])
  const [catalog, setCatalog] = useState<Catalog>()
  const status = useStrixLlamaStatus(s => s.status)
  const pollError = useStrixLlamaStatus(s => s.error)
  const refreshStatus = useStrixLlamaStatus(s => s.refresh)
  const [selected, setSelected] = useState('')
  const [profile, setProfile] = useState<Profile>()
  const [companions, setCompanions] = useState<Companions>()
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState('model')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [noticeKey, setNoticeKey] = useState('')
  const [logText, setLogText] = useState('')
  const [logQuery, setLogQuery] = useState('')
  const [level, setLevel] = useState('all')
  const [paused, setPaused] = useState(false)
  const [follow, setFollow] = useState(true)
  const [rootsText, setRootsText] = useState('')
  const [showRoots, setShowRoots] = useState(false)
  const logCursor = useRef({ offset: 0, file: '' })
  const logBox = useRef<HTMLPreElement>(null)
  const model = catalog?.models.find(m => m.id === selected)
  const tabs = [
    { id: 'models', label: tr('tabs.models'), icon: Database },
    { id: 'configuration', label: tr('tabs.configuration'), icon: SlidersHorizontal },
    { id: 'developer', label: tr('tabs.developer'), icon: Terminal },
  ] as const
  const stateText = (key: string) => tr(`state.${key}`)
  // There is one backend. QSA and Flash Attention are stated rather than offered, because turning
  // either off runs out of memory above 64K context.
  const runtimeName = status?.identity ? status.runtime.split(/[\\/]/).slice(-2, -1)[0] : stateText('notStarted')
  const qsaSupported = model?.architecture === 'qwen4exp'
  const runningQsa = status?.runtime_env?.LLAMA_QSA_SPARSE
  const runningQsaText = runningQsa === '0' ? stateText('off') : runningQsa ? stateText('on') : stateText('unknown')
  const runningNgram = status?.profile ? !!status.profile.ngram_spec : undefined
  const runningNgramText = !status?.identity ? stateText('notLoaded') : runningNgram === undefined ? stateText('unknown') : runningNgram ? stateText('on') : stateText('off')
  const ngramPending = status?.model_path === model?.path && profile && runningNgram !== undefined && profile.ngram_spec !== runningNgram
  const act = async (fn: () => Promise<void>) => {
    setBusy(true); setError(''); setNotice(''); setNoticeKey('')
    try { await fn() } catch (e) { setError(describeError(e, tr)) } finally { setBusy(false) }
  }
  const say = (key: string) => { setNoticeKey(key); setNotice(tr(`notice.${key}`)) }
  const refreshCatalog = async (refresh = false) => {
    const c = await request<Catalog>('catalog', { refresh }); setCatalog(c); setRootsText(c.roots.join('\n'))
    // a rescan may have combined a downloaded draft head with its shared draft; say so, it is the
    // one thing on this page that writes a file
    if (c.merged?.length) setNotice(tr('notice.headMerged', { files: c.merged.map(p => p.split(/[\\/]/).pop()).join(', ') }))
    const failed = Object.entries(c.merge_errors || {})
    if (failed.length) setError(failed.map(([p, e]) => `${p.split(/[\\/]/).pop()}: ${describeError(e, tr)}`).join('\n'))
    return c
  }
  useEffect(() => {
    let disposed = false
    const init = async () => {
      try {
        const c = await request<Catalog>('catalog'); const s = await refreshStatus()
        if (disposed) return
        setCatalog(c); setRootsText(c.roots.join('\n'))
        const remembered = sessionStorage.getItem('strixllama-selected')
        setSelected(c.models.find(m => m.id === remembered && m.role === 'model')?.id || c.models.find(m => m.path === s?.model_path)?.id || c.models.find(m => m.role === 'model')?.id || '')
      } catch (e) { if (!disposed) setError(describeError(e, tr)) }
    }
    void init()
    // the status itself is polled once for the whole app, by StrixLlamaSync
    return () => { disposed = true }
  }, [refreshStatus])
  useEffect(() => {
    let disposed = false
    setProfile(undefined)
    if (selected) sessionStorage.setItem('strixllama-selected', selected)
    if (selected) request<{ profile: Profile; companions: Companions }>('profile', { id: selected }).then(r => { if (!disposed) { setProfile(r.profile); setCompanions(r.companions) } }).catch(e => { if (!disposed) setError(describeError(e, tr)) })
    return () => { disposed = true }
  }, [selected])
  useEffect(() => {
    if (view !== 'developer' || paused) return
    let disposed = false, pending = false
    const poll = async () => {
      if (pending) return; pending = true
      try {
        let r = await request<LogChunk>('logs', { offset: logCursor.current.offset })
        if (disposed) return
        if (r.file !== logCursor.current.file) { logCursor.current = { file: r.file, offset: 0 }; setLogText(''); r = await request<LogChunk>('logs', { offset: 0 }) }
        if (disposed) return
        logCursor.current = { offset: r.offset, file: r.file }
        setLogText(t => ((r.reset ? '' : t) + r.text).split('\n').slice(-2500).join('\n'))
      } catch (e) { if (!disposed) setError(describeError(e, tr)) } finally { pending = false }
    }
    void poll(); const timer = setInterval(poll, 1200)
    return () => { disposed = true; clearInterval(timer) }
  }, [view, paused])
  useEffect(() => { if (follow && logBox.current) logBox.current.scrollTop = logBox.current.scrollHeight }, [logText, follow, logQuery, level])
  useEffect(() => {
    if (status?.status === 'ready' && noticeKey === 'loading') say('loaded')
  }, [status?.status, noticeKey])
  // The provider Jan chats through is kept in step with the server by StrixLlamaSync, app-wide.
  // Here: the one decision the manager takes on its own during a load, which this page must say.
  useEffect(() => {
    if (status?.notice === 'shared_vram_fallback') say('sharedVramFallback')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.notice, status?.identity?.pid])
  const stop = () => act(async () => { await request('stop'); await refreshStatus(); say('unloaded') })
  const start = () => act(async () => { if (!profile) return; await request('start', { id: selected, profile }); await refreshStatus(); say('loading') })
  const save = () => act(async () => { await request('save', { id: selected, profile }); say('saved') })
  const filtered = catalog?.models.filter(m => (kind === 'all' || m.role === kind) && `${m.name} ${m.filename} ${m.architecture} ${m.quant}`.toLowerCase().includes(query.toLowerCase())) || []
  const visibleLogs = logText.split('\n').filter(l => l.toLowerCase().includes(logQuery.toLowerCase()) && (level === 'all' || (level === 'error' ? /\bE\b|ERROR|error:/i.test(l) : /\bW\b|WARN/i.test(l))))
  const field = (key: keyof Profile, value: string | number | boolean) => setProfile(p => p ? { ...p, [key]: value } : p)
  const numeric = (label: string, key: keyof Profile, help: string, min: number, max: number, step = 1) => <label className="space-y-2"><span className="text-sm font-medium">{label}</span><Input type="number" min={min} max={max} step={step} value={String(profile?.[key] ?? '')} onChange={e => field(key, Number(e.target.value))} /><span className="block text-xs text-muted-foreground">{help}</span></label>
  const selectClass = 'rounded-md border bg-background px-3 py-2 text-sm'
  const badge = (text: string) => <span key={text} className="rounded-full border px-2.5 py-1 text-xs text-muted-foreground">{text}</span>
  return <div className="flex h-svh min-h-0 min-w-0 flex-col overflow-hidden pt-12">
    <div className="shrink-0 border-b px-6 pb-4">
      <div className="flex items-center justify-between gap-4"><div><div className="flex items-center gap-3"><h1 className="text-xl font-semibold">{tr('title')}</h1><span className={`rounded-full px-3 py-1 text-xs ${status?.status === 'ready' ? 'bg-emerald-500/10 text-emerald-500' : status?.status === 'loading' ? 'bg-amber-500/10 text-amber-500' : 'bg-muted text-muted-foreground'}`}>{stateText(status?.status || 'stopped')}</span><span className="rounded-full border px-3 py-1 text-xs text-muted-foreground" title={tr('backendHint')}>HIP/ROCm</span></div><p className="mt-1 text-sm text-muted-foreground">{tr('subtitle')}</p></div><div className="flex items-center gap-2"><code className="text-xs text-muted-foreground">{status?.endpoint || 'http://127.0.0.1:8080/v1'}</code><Button size="sm" variant="outline" disabled={busy || !status?.identity} onClick={stop}><Square size={14} />{tr('unload')}</Button></div></div>
      <nav aria-label={tr('nav')} className="mt-5 flex gap-2">{tabs.map(tab => <Link key={tab.id} to={`/strixllama/${tab.id}` as '/strixllama/models'} className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm ${view === tab.id ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:bg-muted'}`}><tab.icon size={16} />{tab.label}</Link>)}</nav>
    </div>
    {(error || pollError) && <div role="alert" className="mx-6 mt-4 rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-500">{error || describeError(pollError, tr)}</div>}
    {status?.failure && !status.identity && <div role="alert" className="mx-6 mt-4 rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-500">{tr('failure', { reason: status.failure_code ? tr(`errors.${status.failure_code}`) : status.failure })}</div>}
    {notice && <div role="status" className="mx-6 mt-4 rounded-lg bg-emerald-500/10 p-3 text-sm text-emerald-600 dark:text-emerald-400">{notice}</div>}
    {view === 'models' && <div className="min-h-0 flex-1 overflow-auto p-6">
      <div className="mb-4 flex flex-wrap items-center gap-3"><div className="relative min-w-64 flex-1"><Search size={16} className="absolute left-3 top-2.5 text-muted-foreground" /><Input aria-label={tr('models.searchLabel')} className="pl-9" placeholder={tr('models.search')} value={query} onChange={e => setQuery(e.target.value)} /></div><select aria-label={tr('models.kindLabel')} className={selectClass} value={kind} onChange={e => setKind(e.target.value)}><option value="model">{tr('models.kindModel')}</option><option value="draft">{tr('models.kindDraft')}</option><option value="projection">{tr('models.kindProjection')}</option><option value="all">{tr('models.kindAll')}</option></select><Button variant="outline" onClick={() => setShowRoots(!showRoots)}>{tr('models.roots')}</Button><Button disabled={busy} variant="outline" onClick={() => act(async () => { await refreshCatalog(true) })}><RefreshCw size={15} />{tr('models.rescan')}</Button></div>
      {showRoots && <div className="mb-4 space-y-3 rounded-lg border p-4"><label className="text-sm">{tr('models.rootsHelp')}<textarea className="mt-2 min-h-20 w-full rounded-md border bg-background p-3 font-mono text-xs" value={rootsText} onChange={e => setRootsText(e.target.value)} /></label><Button disabled={busy} onClick={() => act(async () => { const c = await request<Catalog>('roots', { roots: rootsText.split('\n').map(s => s.trim()).filter(Boolean) }); setCatalog(c); setShowRoots(false) })}>{tr('models.rootsSave')}</Button></div>}
      <div className="overflow-hidden rounded-xl border"><table className="w-full text-sm"><thead className="bg-muted/50 text-xs text-muted-foreground"><tr><th className="p-3 text-left">{tr('models.colModel')}</th><th className="p-3 text-left">{tr('models.colArchitecture')}</th><th className="p-3 text-left">{tr('models.colQuant')}</th><th className="p-3 text-right">{tr('models.colSize')}</th><th className="p-3 text-left">{tr('models.colState')}</th><th className="p-3 text-right">{tr('models.colActions')}</th></tr></thead><tbody>{filtered.map(m => <tr key={m.id} className={`border-t ${selected === m.id ? 'bg-primary/5' : 'hover:bg-muted/30'}`} onClick={() => setSelected(m.id)}><td className="max-w-80 p-3"><div className="truncate font-medium" title={m.name}>{m.name}</div><div className="mt-1 truncate font-mono text-xs text-muted-foreground" title={m.path}>{m.filename}</div></td><td className="p-3 text-muted-foreground">{m.architecture || '—'}</td><td className="p-3"><span className="rounded border px-2 py-0.5 text-xs">{m.quant || '—'}</span></td><td className="whitespace-nowrap p-3 text-right tabular-nums">{gb(m.size)}<div className="text-xs text-muted-foreground">{tr('models.shards', { count: m.shards })}</div></td><td className="p-3 text-xs">{m.error || m.missing?.length ? <span className="text-red-500" title={m.error || m.missing?.join(', ')}>{tr('models.incomplete')}</span> : m.path === status?.model_path && status.identity ? <span className="text-emerald-500">{stateText(status.status)}</span> : tr('models.available')}</td><td className="p-3 text-right"><Link className="text-primary hover:underline" to="/strixllama/configuration" onClick={() => sessionStorage.setItem('strixllama-selected', m.id)}>{tr('models.configure')}</Link></td></tr>)}</tbody></table>{!filtered.length && <div className="p-12 text-center text-muted-foreground">{catalog ? tr('models.empty') : tr('models.loadingCatalog')}</div>}</div>
      <div className="mt-4 flex justify-between text-xs text-muted-foreground"><span>{tr('models.summary', { count: filtered.length, size: gb(filtered.reduce((n,m) => n+m.size,0)) })}</span><span>{tr('models.readOnly')}</span></div>
    </div>}
    {view === 'configuration' && <div className="min-h-0 flex-1 overflow-auto p-6"><div className="mb-5 flex items-center gap-3"><select aria-label={tr('config.targetLabel')} className={`${selectClass} min-w-0 flex-1`} value={selected} onChange={e => setSelected(e.target.value)}>{catalog?.models.filter(m => m.role === 'model').map(m => <option key={m.id} value={m.id}>{m.filename}</option>)}</select><Button variant="outline" disabled={busy || !profile} onClick={save}>{tr('config.save')}</Button><Button disabled={busy || !profile || !!status?.identity} onClick={start}><Play size={15} />{tr('config.load')}</Button></div>
      {model && <div className="mb-5 rounded-xl border bg-muted/20 p-4">
        <div className="font-medium">{model.name}</div>
        <p className="mt-2 break-all font-mono text-xs text-muted-foreground">{model.path}</p>
        <p className="mt-2 text-xs text-muted-foreground">{model.architecture} · {model.quant} · {gb(model.size)} · {tr('config.declaredContext', { context: model.context?.toLocaleString() || tr('config.contextUnknown') })}</p>
        {qsaSupported && <>
          <div className="mt-3 flex flex-wrap gap-2">{[tr('config.runtime.flashAttention'), tr('config.runtime.qsa'), tr('config.runtime.kv'), tr('config.runtime.weights')].map(badge)}</div>
          <p className="mt-2 text-xs text-muted-foreground">{tr('config.runtime.note')}</p>
        </>}
      </div>}
      {profile && <div className="grid gap-5 xl:grid-cols-2">
        <section className="rounded-xl border p-5">
          <h2 className="mb-5 font-medium">{tr('config.generation.title')}</h2>
          <label className="flex items-center justify-between gap-4 text-sm"><span className="font-medium">{tr('config.generation.thinking')}</span><select aria-label={tr('config.generation.thinking')} className={selectClass} value={String(profile.thinking)} onChange={e => field('thinking', e.target.value)}>{THINKING_LEVELS.map(l => <option key={l} value={l}>{tr('config.generation.thinkingLevel.' + l)}</option>)}</select></label>
          <p className="mt-2 text-xs text-muted-foreground">{tr('config.generation.thinkingHelp.' + String(profile.thinking))}</p>
          <p className="mt-2 text-xs text-muted-foreground">{tr('config.generation.thinkingNote')}</p>
          <h2 className="mb-5 mt-6 border-t pt-5 font-medium">{tr('config.context.title')}</h2>
          <div className="grid grid-cols-2 gap-5">
            {numeric(tr('config.context.length'), 'context', tr('config.context.lengthHelp'), 512, model?.context || 262144)}
            {numeric(tr('config.context.gpuLayers'), 'gpu_layers', tr('config.context.gpuLayersHelp'), 0, 999)}
          </div>
        </section>
        <section className="rounded-xl border p-5">
          <h2 className="mb-5 font-medium">{tr('config.speculative.title')}</h2>
          <label className="flex items-center justify-between gap-4 text-sm"><span className="font-medium">{tr('config.speculative.mtp')}</span><Switch id="mtp" checked={profile.mtp} disabled={!qsaSupported} onCheckedChange={v => field('mtp', v)} /></label>
          {!qsaSupported && <p className="mt-2 text-xs text-muted-foreground">{tr('config.speculative.mtpUnsupported')}</p>}
          {qsaSupported && companions && !companions.draft && <p className="mt-2 text-xs text-amber-500">{tr('config.speculative.noDraft')}</p>}
          <label className="mt-4 block text-sm">{tr('config.speculative.draftModel')}<select className={`${selectClass} mt-2 w-full`} value={profile.draft} onChange={e => field('draft', e.target.value)} disabled={!profile.mtp}>{/* a saved draft outside the scanned directories would otherwise render as whichever option comes first */}{profile.draft && !catalog?.models.some(m => m.role === 'draft' && m.path === profile.draft) && <option value={profile.draft}>{profile.draft.split(/[\\/]/).pop()} — {tr('config.speculative.draftMissing')}</option>}{catalog?.models.filter(m => m.role === 'draft').map(m => <option key={m.id} value={m.path}>{m.filename}</option>)}</select></label>
          {companions?.draft && !companions.draft_head && <p className="mt-2 text-xs text-muted-foreground">{tr('config.speculative.draftHelp')}</p>}
          <div className="mt-5 grid grid-cols-2 gap-5">
            {numeric(tr('config.speculative.draftMax'), 'draft_max', tr('config.speculative.draftMaxHelp'), 1, 8)}
            {numeric(tr('config.speculative.draftMin'), 'draft_min', tr('config.speculative.draftMinHelp'), 0, 1, 0.05)}
          </div>
          <h2 className="mb-5 mt-6 border-t pt-5 font-medium">{tr('config.vision.title')}</h2>
          <label className="flex items-center justify-between gap-4 text-sm"><span className="font-medium">{tr('config.vision.toggle')}</span><Switch id="vision" checked={!!profile.vision} onCheckedChange={v => field('vision', v)} /></label>
          <p className="mt-2 text-xs text-muted-foreground">{tr('config.vision.help')}</p>
          <p className="mt-2 text-xs text-muted-foreground">{tr('config.vision.cost')}</p>
          {companions && !companions.mmproj && !profile.mmproj && <p className="mt-2 text-xs text-amber-500">{tr('config.vision.noProjector')}</p>}
          {catalog?.models.some(m => m.role === 'projection') && <label className="mt-4 block text-sm">{tr('config.vision.projector')}<select className={`${selectClass} mt-2 w-full`} value={profile.mmproj} onChange={e => field('mmproj', e.target.value)} disabled={!profile.vision}><option value="">{tr('config.vision.projectorAuto')}</option>{catalog?.models.filter(m => m.role === 'projection').map(m => <option key={m.id} value={m.path}>{m.filename}</option>)}</select></label>}
        </section>
      </div>}
      {profile && <details className="mt-5 rounded-xl border p-5">
        <summary className="cursor-pointer text-sm font-medium">{tr('config.advanced.summary')}</summary>
        <p className="mt-3 text-xs text-muted-foreground">{tr('config.advanced.note')}</p>
        <div className="mt-5 grid gap-5 xl:grid-cols-2">
          <div className="grid grid-cols-2 gap-5">
            {numeric(tr('config.advanced.batch'), 'batch', tr('config.advanced.batchHelp'), 32, 32768)}
            {numeric(tr('config.advanced.ubatch'), 'ubatch', tr('config.advanced.ubatchHelp'), 32, 32768)}
            {numeric(tr('config.advanced.threads'), 'threads', tr('config.advanced.threadsHelp'), 1, 32)}
            {numeric(tr('config.advanced.parallel'), 'parallel', tr('config.advanced.parallelHelp'), 1, 8)}
          </div>
          <div className="space-y-5">
            <div>
              <label className="flex items-center justify-between gap-4 text-sm"><span className="font-medium">{tr('config.advanced.ngram')}</span><Switch id="ngram-spec" checked={!!profile.ngram_spec} onCheckedChange={v => field('ngram_spec', v)} /></label>
              <p className="mt-2 text-xs text-muted-foreground">{tr('config.advanced.ngramHelp')}</p>
              <p className="mt-2 text-xs text-muted-foreground">{tr('config.advanced.ngramRunning', { state: runningNgramText })} {ngramPending ? tr('config.advanced.ngramPending') : tr('config.advanced.applyOnReload')}</p>
            </div>
          </div>
        </div>
      </details>}
      <p className="mt-5 text-sm text-muted-foreground">{tr('config.footer')}</p>
    </div>}
    {view === 'developer' && <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-hidden p-6"><div className="grid grid-cols-3 gap-3"><div className="rounded-lg border p-3 text-sm"><span className="text-xs text-muted-foreground">{tr('developer.process')}</span><div className="mt-1 font-mono">{status?.identity?.pid || '—'}</div></div><div className="rounded-lg border p-3 text-sm"><span className="text-xs text-muted-foreground">{tr('developer.backend')}</span><div className="mt-1">{runtimeName}{status?.identity ? ` · QSA ${runningQsaText}` : ''}</div></div><div className="rounded-lg border p-3 text-sm"><span className="text-xs text-muted-foreground">{tr('developer.state')}</span><div className="mt-1">{stateText(status?.status || 'stopped')}</div></div></div><details className="max-h-40 shrink-0 overflow-auto rounded-lg border p-3 text-xs"><summary className="cursor-pointer text-muted-foreground">{tr('developer.launchParams')}</summary><pre className="mt-3 whitespace-pre-wrap break-all">{status?.command || tr('developer.notStarted')}{status?.runtime_env && `\n\n${Object.entries(status.runtime_env).map(([key, value]) => `${key}=${value}`).join('\n')}`}</pre></details><div className="flex flex-wrap gap-2"><Input aria-label={tr('developer.searchLogsLabel')} className="min-w-40 flex-1" placeholder={tr('developer.searchLogs')} value={logQuery} onChange={e => setLogQuery(e.target.value)} /><select className={selectClass} value={level} onChange={e => setLevel(e.target.value)} aria-label={tr('developer.levelLabel')}><option value="all">{tr('developer.levelAll')}</option><option value="warning">{tr('developer.levelWarning')}</option><option value="error">{tr('developer.levelError')}</option></select><Button variant="outline" onClick={() => setPaused(!paused)}>{paused ? tr('developer.resume') : tr('developer.pause')}</Button><Button variant="outline" onClick={() => { void navigator.clipboard.writeText(visibleLogs.join('\n')).catch(e => setError(describeError(e, tr))) }}><Copy size={14} />{tr('developer.copy')}</Button><Button variant="outline" onClick={() => { const url=URL.createObjectURL(new Blob([visibleLogs.join('\n')],{type:'text/plain;charset=utf-8'})); const a=document.createElement('a');a.href=url;a.download='strixllama-developer.log';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000) }}><Download size={14} />{tr('developer.export')}</Button></div><pre ref={logBox} tabIndex={0} role="log" aria-label={tr('developer.logLabel')} aria-live="off" onScroll={e => { const box = e.currentTarget; setFollow(box.scrollHeight - box.clientHeight - box.scrollTop < 24) }} className="strixllama-logs min-h-0 min-w-0 flex-1 rounded-xl border bg-zinc-950 p-4 font-mono text-xs leading-5 text-zinc-300">{visibleLogs.length && logText ? visibleLogs.map((l,i) => <div key={i} className={/\bE\b|ERROR/i.test(l) ? 'text-red-400' : /\bW\b|WARN/i.test(l) ? 'text-amber-400' : /tokens per second|acceptance|tg =/.test(l) ? 'text-emerald-400' : ''}>{l || ' '}</div>) : tr('developer.waiting')}</pre><div className="flex justify-between text-xs text-muted-foreground"><span className="truncate pr-4">{logCursor.current.file}</span><label className="flex shrink-0 items-center gap-2"><input type="checkbox" checked={follow} onChange={e => setFollow(e.target.checked)} />{tr('developer.autoscroll')}</label></div></div>}
  </div>
}
