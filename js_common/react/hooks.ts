import { useState, useEffect } from 'react'
import { Maybe } from '../types'
import axios from 'axios'

const POLL_INTERVAL_MILLIS = 1000

export function usePollingApi<T>(apiPath: string): Maybe<T> {
  const [value, setValue] = useState<Maybe<T>>(undefined)

  const fetchData = () => axios.get(apiPath).then(res => setValue(res.data))
  useEffect(() => {
    fetchData()
    const pollInterval = setInterval(fetchData, POLL_INTERVAL_MILLIS)

    return () => clearInterval(pollInterval)
  }, [])

  return value
}

export function useStreamingApi<T>(apiPath: string): Maybe<T> {
  const [value, setValue] = useState<Maybe<T>>(undefined)

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:5050${apiPath}`)
    ws.onmessage = ev => setValue(JSON.parse(ev.data))

    return () => ws.close()
  }, [])

  return value
}
