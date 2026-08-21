import { useCallback, useEffect, useState } from 'react'
import { api, type FiltersResponse, type ShadowbanLevel } from '../api'
import { Panel } from '../components/Stat'

const LEVELS: ShadowbanLevel[] = ['Clean', 'Monitoring', 'Suppressed', 'Ghosted']

export function Moderation({ apiKey }: { apiKey: string }) {
  const [filters, setFilters] = useState<FiltersResponse | null>(null)
  const [message, setMessage] = useState<{ text: string; tone: 'positive' | 'negative' } | null>(null)

  const [sbUserId, setSbUserId] = useState('')
  const [sbLevel, setSbLevel] = useState<ShadowbanLevel>('Monitoring')
  const [sbReason, setSbReason] = useState('')
  const [sbExpires, setSbExpires] = useState('')
  const [sbBusy, setSbBusy] = useState(false)

  const [banUserId, setBanUserId] = useState('')
  const [banReason, setBanReason] = useState('')
  const [banBusy, setBanBusy] = useState(false)

  const load = useCallback(async () => {
    setFilters(await api.filters(apiKey))
  }, [apiKey])

  useEffect(() => {
    load()
  }, [load])

  async function submitShadowban(e: React.FormEvent) {
    e.preventDefault()
    if (!sbUserId.trim()) return
    setSbBusy(true)
    setMessage(null)
    try {
      const res = await api.setShadowban(
        apiKey,
        sbUserId.trim(),
        sbLevel,
        sbReason.trim(),
        sbExpires.trim() ? Number(sbExpires) : null,
      )
      setMessage({ text: res.message, tone: 'positive' })
      setSbUserId('')
      setSbReason('')
      setSbExpires('')
      await load()
    } catch (err) {
      setMessage({ text: err instanceof Error ? err.message : 'Échec.', tone: 'negative' })
    } finally {
      setSbBusy(false)
    }
  }

  async function submitBan(e: React.FormEvent) {
    e.preventDefault()
    if (!banUserId.trim()) return
    setBanBusy(true)
    setMessage(null)
    try {
      const res = await api.ban(apiKey, banUserId.trim(), banReason.trim())
      setMessage({ text: res.message, tone: 'positive' })
      setBanUserId('')
      setBanReason('')
      await load()
    } catch (err) {
      setMessage({ text: err instanceof Error ? err.message : 'Échec.', tone: 'negative' })
    } finally {
      setBanBusy(false)
    }
  }

  async function unban(userId: string) {
    setMessage(null)
    try {
      const res = await api.unban(apiKey, userId)
      setMessage({ text: res.message, tone: 'positive' })
      await load()
    } catch (err) {
      setMessage({ text: err instanceof Error ? err.message : 'Échec.', tone: 'negative' })
    }
  }

  return (
    <div className="dashboard">
      {message && <p className={`form-message tone-${message.tone}`}>{message.text}</p>}

      <div className="grid-2">
        <Panel title="Régler la visibilité (shadowban)">
          <form className="form-stack" onSubmit={submitShadowban}>
            <label>
              Identifiant utilisateur
              <input value={sbUserId} onChange={(e) => setSbUserId(e.target.value)} placeholder="uuid" required />
            </label>
            <label>
              Niveau
              <select value={sbLevel} onChange={(e) => setSbLevel(e.target.value as ShadowbanLevel)}>
                {LEVELS.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Raison
              <input value={sbReason} onChange={(e) => setSbReason(e.target.value)} placeholder="optionnel" />
            </label>
            <label>
              Expire dans (jours)
              <input
                type="number"
                min={1}
                value={sbExpires}
                onChange={(e) => setSbExpires(e.target.value)}
                placeholder="vide = sans terme"
              />
            </label>
            <button className="button-primary" type="submit" disabled={sbBusy}>
              {sbBusy ? 'Application…' : 'Appliquer'}
            </button>
          </form>
        </Panel>

        <Panel title="Bannissement définitif">
          <form className="form-stack" onSubmit={submitBan}>
            <label>
              Identifiant utilisateur
              <input value={banUserId} onChange={(e) => setBanUserId(e.target.value)} placeholder="uuid" required />
            </label>
            <label>
              Raison
              <input value={banReason} onChange={(e) => setBanReason(e.target.value)} placeholder="optionnel" />
            </label>
            <button className="button-danger" type="submit" disabled={banBusy}>
              {banBusy ? 'Application…' : 'Bannir'}
            </button>
          </form>
        </Panel>
      </div>

      <Panel title={`Sous surveillance (${filters?.total_shadowbanned ?? 0})`}>
        {!filters || filters.shadowbanned.length === 0 ? (
          <p className="panel-note">Aucun compte.</p>
        ) : (
          <table className="list-table mono">
            <tbody>
              {filters.shadowbanned.map((u) => (
                <tr key={u.user_id}>
                  <td>{u.user_id}</td>
                  <td>{u.level}</td>
                  <td className="dim-name">{u.reason ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      <Panel title={`Bannis (${filters?.total_hard_banned ?? 0})`}>
        {!filters || filters.hard_banned.length === 0 ? (
          <p className="panel-note">Aucun compte.</p>
        ) : (
          <table className="list-table mono">
            <tbody>
              {filters.hard_banned.map((u) => (
                <tr key={u.user_id}>
                  <td>{u.user_id}</td>
                  <td className="dim-name">{u.reason ?? '—'}</td>
                  <td>
                    <button onClick={() => unban(u.user_id)}>Débannir</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  )
}
