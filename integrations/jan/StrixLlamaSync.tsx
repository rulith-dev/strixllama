import { useEffect } from 'react'
import cloneDeep from 'lodash/cloneDeep'
import { useModelProvider } from '@/hooks/useModelProvider'
import { openAIProviderSettings } from '@/constants/providers'
import { localStorageKey } from '@/constants/localStorage'
import { ENDPOINT, PROVIDER, useStrixLlamaStatus } from './status'

// Mounted once at the root. Polls the manager, and keeps the chat side of Jan pointed at it.
//
// Jan's chat talks to whatever it calls a provider. This build has exactly one, the local server
// tools/manager.py starts, so it is registered here on first run instead of asking for a name, a
// URL and a key that could only ever be one thing. Its model list follows the running server:
// what /v1/models reports, shown under the name the catalog has for the file. Nothing here is
// specific to this machine: the endpoint is the manager's fixed loopback port.
export function StrixLlamaSync() {
  const refresh = useStrixLlamaStatus((s) => s.refresh)
  const status = useStrixLlamaStatus((s) => s.status)
  const providers = useModelProvider((s) => s.providers)

  useEffect(() => {
    let pending = false
    const tick = async () => {
      if (pending) return
      pending = true
      try { await refresh() } finally { pending = false }
    }
    void tick()
    const timer = setInterval(tick, 3000)
    return () => clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    // Wait for Jan's own list before adding to it: the persisted store hydrates over whatever is
    // in memory, and an entry added before that would be lost and added twice.
    if (!providers.length || providers.some((p) => p.provider === PROVIDER)) return
    const settings = cloneDeep(openAIProviderSettings) as ProviderSetting[]
    for (const s of settings) {
      if (s.key === 'base-url') (s.controller_props as { value?: string }).value = ENDPOINT
      if (s.key === 'api-key') (s.controller_props as { value?: string }).value = ''
    }
    useModelProvider.getState().addProvider({
      provider: PROVIDER,
      active: true,
      models: [],
      settings,
      base_url: ENDPOINT,
      api_key: '',
    })
  }, [providers])

  useEffect(() => {
    if (status?.status !== 'ready' || !status.served_models?.length) return
    const p = providers.find((p) => p.provider === PROVIDER)
    if (!p) return
    const models = status.served_models.map((m) => ({
      ...p.models.find((old) => old.id === m.id),
      id: m.id,
      displayName: status.model_name || m.id,
    }))
    if (JSON.stringify(p.models) !== JSON.stringify(models)) {
      useModelProvider.getState().updateProvider(PROVIDER, { models })
    }
    // Nothing chosen yet (a fresh install, or the picker mounted before the server answered):
    // pick the model that is actually loaded, and make it the one new chats start with.
    const store = useModelProvider.getState()
    if (!store.selectedModel && models.length) {
      store.selectModelProvider(PROVIDER, models[0].id)
      localStorage.setItem(localStorageKey.lastUsedModel, JSON.stringify({ provider: PROVIDER, model: models[0].id }))
    }
  }, [status, providers])

  return null
}
