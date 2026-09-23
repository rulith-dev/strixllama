import { create } from 'zustand'
import { invoke } from '@tauri-apps/api/core'

// One IPC round trip to tools/manager.py. The helper reads a JSON request on stdin and answers on
// stdout; there is no network port and no shell in between.
export const request = <T,>(op: string, data: object = {}) =>
  invoke<T>('strixllama_request', { request: { op, data } })

// The provider Jan's chat side talks to. There is exactly one in this build: the local server the
// manager starts, on the endpoint below. StrixLlamaSync registers it on first run.
export const PROVIDER = 'strixllama'
export const ENDPOINT = 'http://127.0.0.1:8080/v1'

export type Profile = { thinking: string; context: number; gpu_layers: number; threads: number; batch: number; ubatch: number; mtp: boolean; draft: string; draft_max: number; draft_min: number; ngram_spec: boolean; kv: string; flash_attention: string; qsa: boolean; shared_vram: boolean; trunk_decode_q6k: boolean; parallel: number; kv_pool: number; vision: boolean; mmproj: string; prompt_cache_disk: boolean; prompt_cache_disk_mib: number }
export type Status = {
  status: string; endpoint: string; runtime: string; runtime_available?: boolean; runtime_env?: Record<string, string>
  identity?: { pid: number }; model_path?: string; model_name?: string; log?: string; command?: string; adopted?: boolean; profile?: Profile
  served_models?: { id: string }[]
  // set by the manager: whether this load runs in shared GPU memory, the dedicated carve it saw,
  // a one-off notice about the load (shared_vram_fallback), and why the last load ended if it
  // ended badly
  unified?: boolean; dedicated_vram?: number | null; notice?: string; failure?: string; failure_code?: string
  // a ready server: Windows' commit limit (RAM + page file) and what is left of it, and whether that is too little
  commit_limit?: number; commit_available?: number; commit_low?: boolean
}

// A failed request arrives as the manager's whole error record, {error, code, params}, when it
// has a code (strixllama.rs passes it through as JSON), so the page can render `errors.<code>`
// from its own locale with the parameters filled in; the English text is the fallback. Anything
// else - the Rust side, the IPC - is a plain string.
export const describeError = (e: unknown, tr: (key: string, options?: Record<string, unknown>) => string) => {
  const text = String(e)
  if (text.startsWith('{')) {
    try {
      const parsed = JSON.parse(text)
      if (parsed && typeof parsed.code === 'string')
        return tr(`errors.${parsed.code}`, { defaultValue: String(parsed.error || parsed.code), ...(parsed.params || {}) })
    } catch { /* not the manager's record */ }
  }
  return text
}

type State = { status?: Status; error: string; refresh: () => Promise<Status | undefined> }

// The manager's status, polled once for the whole app by StrixLlamaSync and read by the pages, so
// the three management views and the provider sync share one poll instead of each running its own.
export const useStrixLlamaStatus = create<State>((set) => ({
  status: undefined,
  error: '',
  refresh: async () => {
    try {
      const status = await request<Status>('status')
      set({ status, error: '' })
      return status
    } catch (e) {
      set({ error: String(e) })
      return undefined
    }
  },
}))
