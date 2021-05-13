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

// TODO This closes the websocket when the component using this
// hook is unmounted. It is not clear to me what the best overall
// strategy is around websockets, and whether we want to prioritize
// simple 1:1 socket to REST url or minimize number of opening and
// closing connections with a more complicated method of passing
// messages.
export function useStreamingApi<T>(apiPath: string): Maybe<T> {
  const [value, setValue] = useState<Maybe<T>>(undefined)

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:5050${apiPath}`)
    ws.onmessage = ev => setValue(JSON.parse(ev.data))

    return () => ws.close()
  }, [])

  return value
}
