import { useState, type FormEvent } from 'react'
import { api, ApiError } from './api'

export function SignIn({ onSignIn }: { onSignIn: (key: string) => void }) {
  const [value, setValue] = useState('')
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!value.trim()) return
    setChecking(true)
    setError(null)
    try {
      await api.stats(value.trim())
      onSignIn(value.trim())
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? 'Clé refusée.' : "Le service ne répond pas.")
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="signin">
      <form className="signin-card" onSubmit={handleSubmit}>
        <div className="signin-mark">NR</div>
        <h1>NeuralRank Console</h1>
        <p className="signin-sub">Clé d'administration du recommandeur</p>
        <input
          type="password"
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="X-Admin-Key"
          spellCheck={false}
          autoComplete="off"
        />
        {error && <div className="signin-error">{error}</div>}
        <button type="submit" disabled={checking || !value.trim()}>
          {checking ? 'Vérification…' : 'Entrer'}
        </button>
      </form>
    </div>
  )
}
