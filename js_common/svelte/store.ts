import { writable, Writable } from 'svelte/store'
import type { Maybe } from '../types'
import axios from 'axios'

const POLL_INTERVAL_MILLIS = 1000

export type StoreApiResult<T> = {
  store: Writable<Maybe<T>>,
  destroy: () => void,
}

export function pollingApiStore<T>(apiPath: string): StoreApiResult<T> {
  const store = writable(undefined)
  const fetchData = () => axios.get(apiPath).then(res => store.set(res.data))
  fetchData()
  const interval = setInterval(fetchData, POLL_INTERVAL_MILLIS)

  return { store, destroy: () => clearInterval(interval) }
}

export function streamingApiStore<T>(apiPath: string): StoreApiResult<T> {
  const store = writable(undefined)
  const ws = new WebSocket(`ws://localhost:5050${apiPath}`)
  ws.onmessage = ev => store.set(JSON.parse(ev.data))

  return { store, destroy: () => ws.close() }
}
