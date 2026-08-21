import { useCallback, useEffect, useState } from 'react'
import { api, DIMENSION_LABELS, DIMENSION_ORDER, type AlgoWeights } from '../api'
import { Panel, Pill } from '../components/Stat'
import { WeightBars } from '../components/WeightBars'

const KEY_TO_FIELD: Record<keyof AlgoWeights, keyof import('../api').SetWeightsRequest> = {
  d1_engagement_velocity: 'd1',
  d2_content_intelligence: 'd2',
  d3_social_graph: 'd3',
  d4_temporal: 'd4',
  d5_behavioral: 'd5',
  d6_diversity: 'd6',
  d7_viral: 'd7',
  d8_personalization: 'd8',
  d9_llm_understanding: 'd9',
}

export function Weights({ apiKey }: { apiKey: string }) {
  const [weights, setWeights] = useState<AlgoWeights | null>(null)
  const [autoTuned, setAutoTuned] = useState(false)
  const [draft, setDraft] = useState<AlgoWeights | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ text: string; tone: 'positive' | 'negative' } | null>(null)

  const load = useCallback(async () => {
    const res = await api.weights(apiKey)
    setWeights(res.weights)
    setAutoTuned(res.auto_tuned)
    setDraft(res.weights)
  }, [apiKey])

  useEffect(() => {
    load()
  }, [load])

  const sum = draft ? DIMENSION_ORDER.reduce((s, k) => s + draft[k], 0) : 0
  const dirty = draft && weights && DIMENSION_ORDER.some((k) => draft[k] !== weights[k])

  async function save() {
    if (!draft) return
    setSaving(true)
    setMessage(null)
    try {
      const partial: Record<string, number> = {}
      for (const k of DIMENSION_ORDER) partial[KEY_TO_FIELD[k]] = draft[k]
      const res = await api.setWeights(apiKey, partial)
      setMessage({ text: res.message, tone: 'positive' })
      await load()
    } catch (err) {
      setMessage({ text: err instanceof Error ? err.message : 'Échec de la sauvegarde.', tone: 'negative' })
    } finally {
      setSaving(false)
    }
  }

  async function resetToAuto() {
    setSaving(true)
    setMessage(null)
    try {
      const res = await api.resetWeights(apiKey)
      setMessage({ text: res.message, tone: 'positive' })
      await load()
    } catch (err) {
      setMessage({ text: err instanceof Error ? err.message : 'Échec de la réinitialisation.', tone: 'negative' })
    } finally {
      setSaving(false)
    }
  }

  if (!weights || !draft) return <div className="loading">Lecture des poids…</div>

  return (
    <div className="dashboard">
      <div className="hero-status">
        <Pill tone={autoTuned ? 'positive' : 'idle'}>{autoTuned ? "Piloté par l'auto-tuner" : 'Réglage manuel actif'}</Pill>
      </div>
      <p className="panel-note">
        Modifier une valeur ci-dessous fige les poids : l'auto-tuner s'arrête d'ajuster tant qu'un réglage manuel est actif.
        « Rendre à l'auto-tuner » efface ce réglage et le laisse reprendre la main.
      </p>

      <Panel title="Aperçu">
        <WeightBars weights={draft} compare={dirty ? weights : undefined} compareLabel="actif" />
      </Panel>

      <Panel
        title="Contrôles"
        action={
          <div className="button-row">
            <button onClick={resetToAuto} disabled={saving}>
              Rendre à l'auto-tuner
            </button>
            <button className="button-primary" onClick={save} disabled={saving || !dirty}>
              {saving ? 'Sauvegarde…' : 'Sauvegarder'}
            </button>
          </div>
        }
      >
        <div className="weight-sum mono">
          Somme : {sum.toFixed(3)}
          {Math.abs(sum - 1) > 0.01 && <span className="tone-negative"> — s'écarte de 1.0</span>}
        </div>
        <div className="sliders">
          {DIMENSION_ORDER.map((key) => (
            <div className="slider-row" key={key}>
              <label htmlFor={key}>
                <span className="mono">{DIMENSION_LABELS[key].short}</span> {DIMENSION_LABELS[key].name}
              </label>
              <input
                id={key}
                type="range"
                min={0}
                max={0.5}
                step={0.005}
                value={draft[key]}
                onChange={(e) => setDraft({ ...draft, [key]: Number(e.target.value) })}
              />
              <span className="slider-value mono">{(draft[key] * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        {message && <p className={`form-message tone-${message.tone}`}>{message.text}</p>}
      </Panel>
    </div>
  )
}
