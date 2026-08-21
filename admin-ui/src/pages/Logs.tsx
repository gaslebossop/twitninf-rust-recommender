import { useCallback, useEffect, useState } from 'react'
import { api, type LogEntry } from '../api'
import { Panel } from '../components/Stat'

export function Logs({ apiKey }: { apiKey: string }) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.logs(apiKey)
      setLogs(res.logs)
    } finally {
      setLoading(false)
    }
  }, [apiKey])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div className="dashboard">
      <Panel
        title="Journal du service (100 dernières lignes)"
        action={
          <button onClick={load} disabled={loading}>
            {loading ? 'Lecture…' : 'Actualiser'}
          </button>
        }
      >
        <div className="log-stream mono">
          {logs.map((l, i) => (
            <div className={`log-line log-${l.level.toLowerCase()}`} key={i}>
              <span className="log-ts">{l.timestamp}</span>
              <span className="log-level">{l.level}</span>
              <span className="log-msg">{l.message}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )
}
