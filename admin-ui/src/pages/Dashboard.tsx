import { useCallback, useEffect, useState } from 'react'
import { api, DIMENSION_LABELS, DIMENSION_ORDER, type AlgoStats, type AlgoWeights, type BackfillReport } from '../api'
import { Panel, Pill, Stat } from '../components/Stat'
import { WeightBars } from '../components/WeightBars'

const REFRESH_MS = 20_000

function pct(n: number, digits = 1) {
  return `${(n * 100).toFixed(digits)}%`
}

function arrayToWeights(arr: number[]): AlgoWeights {
  // `resulting_weights` du rattrapage suit l'ordre d'`extract_features` :
  // les 8 dimensions puis 7 traits de contexte que cette page n'affiche pas.
  const [d1, d2, d3, d4, d5, d6, d7, d8] = arr
  return {
    d1_engagement_velocity: d1,
    d2_content_intelligence: d2,
    d3_social_graph: d3,
    d4_temporal: d4,
    d5_behavioral: d5,
    d6_diversity: d6,
    d7_viral: d7,
    d8_personalization: d8,
    d9_llm_understanding: 0,
  }
}

export function Dashboard({ apiKey }: { apiKey: string }) {
  const [stats, setStats] = useState<AlgoStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const [sinceDays, setSinceDays] = useState(14)
  const [report, setReport] = useState<BackfillReport | null>(null)
  const [running, setRunning] = useState<'dry' | 'apply' | null>(null)
  const [backfillError, setBackfillError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const s = await api.stats(apiKey)
      setStats(s)
      setLastUpdated(new Date())
      setError(null)
    } catch {
      setError('Lecture impossible — le service ne répond pas.')
    }
  }, [apiKey])

  useEffect(() => {
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => clearInterval(id)
  }, [load])

  async function runDryRun() {
    setRunning('dry')
    setBackfillError(null)
    try {
      const res = await api.backfillCtr(apiKey, sinceDays, false)
      setReport(res.report)
    } catch {
      setBackfillError('Le rattrapage a échoué — voir les logs.')
    } finally {
      setRunning(null)
    }
  }

  async function applyBackfill() {
    if (!report) return
    setRunning('apply')
    setBackfillError(null)
    try {
      const res = await api.backfillCtr(apiKey, sinceDays, true)
      setReport(res.report)
      await load()
    } catch {
      setBackfillError("L'application a échoué — voir les logs. Une sauvegarde n'a peut-être pas été écrite.")
    } finally {
      setRunning(null)
    }
  }

  if (error && !stats) {
    return (
      <div className="empty-state">
        <p>{error}</p>
        <button onClick={load}>Réessayer</button>
      </div>
    )
  }

  if (!stats) {
    return <div className="loading">Lecture des instruments…</div>
  }

  return (
    <div className="dashboard">
      <div className="hero">
        <div className="hero-status">
          <Pill tone={stats.auto_tuned ? 'positive' : 'idle'}>
            {stats.auto_tuned ? 'Auto-réglé' : 'Poids par défaut'}
          </Pill>
          <Pill tone={stats.ml_active ? 'accent' : 'idle'}>{stats.ml_active ? 'CTR appris actif' : 'CTR en amorçage'}</Pill>
          <Pill tone={stats.dwell_active ? 'accent' : 'idle'}>
            {stats.dwell_active ? 'Dwell appris actif' : 'Dwell en amorçage'}
          </Pill>
        </div>
        <h1>{stats.algorithm_version}</h1>
        {lastUpdated && (
          <p className="hero-meta mono">
            actualisé il y a {Math.max(0, Math.round((Date.now() - lastUpdated.getTime()) / 1000))}s
          </p>
        )}
      </div>

      <div className="grid-3">
        <Panel title="Modèle de clic (CTR)">
          <div className="stat-row">
            <Stat label="Échantillons" value={stats.ctr_samples.toLocaleString('fr-FR')} />
            <Stat label="Taux global" value={pct(stats.global_ctr, 2)} />
          </div>
          <p className="panel-note">
            {stats.ml_active
              ? 'Mélangé à 40% dans le classement (seuil de 200 échantillons franchi).'
              : `${200 - stats.ctr_samples} échantillons avant activation.`}
          </p>
        </Panel>

        <Panel title="Modèle de temps passé (dwell)">
          <div className="stat-row">
            <Stat label="Échantillons" value={stats.dwell_samples.toLocaleString('fr-FR')} />
            <Stat
              label="Poids moyen observé"
              value={stats.dwell_mean_weight >= 0 ? `+${stats.dwell_mean_weight.toFixed(3)}` : stats.dwell_mean_weight.toFixed(3)}
              tone={stats.dwell_mean_weight >= 0 ? 'positive' : 'negative'}
            />
          </div>
          <p className="panel-note">
            {stats.dwell_active
              ? 'Mélangé dans le classement (seuil de 200 échantillons franchi).'
              : `${Math.max(0, 200 - stats.dwell_samples)} échantillons avant activation.`}
          </p>
        </Panel>

        <Panel title="Auto-tuner">
          <div className="stat-row">
            <Stat label="État" value={stats.auto_tuned ? 'Actif' : 'Amorçage'} tone={stats.auto_tuned ? 'positive' : 'idle'} />
          </div>
          <p className="panel-note">
            {stats.auto_tuned
              ? "Les poids ci-dessous viennent du modèle CTR, pas des valeurs par défaut."
              : "Redémarre les poids par défaut à chaque redémarrage du service, jusqu'à 500 échantillons CTR ET 100 nouveaux depuis le dernier réglage."}
          </p>
        </Panel>
      </div>

      <Panel title="Poids des 9 dimensions">
        <WeightBars weights={stats.weights} compare={report ? arrayToWeights(report.resulting_weights) : undefined} compareLabel="rattrapage" />
      </Panel>

      <Panel
        title="Rattrapage du modèle CTR"
        action={
          <label className="days-input">
            <span>Fenêtre</span>
            <input
              type="number"
              min={1}
              max={90}
              value={sinceDays}
              onChange={(e) => setSinceDays(Math.max(1, Math.min(90, Number(e.target.value) || 14)))}
            />
            <span>jours</span>
          </label>
        }
      >
        <p className="panel-note">
          Reconstruit le modèle depuis les interactions réelles de la fenêtre — utile quand les poids actuels reflètent un
          historique jugé peu représentatif (voir D6 par exemple). Toujours lancer un essai avant d'appliquer : l'essai ne
          modifie rien sur disque.
        </p>
        <div className="button-row">
          <button onClick={runDryRun} disabled={running !== null}>
            {running === 'dry' ? 'Reconstruction…' : "Lancer l'essai"}
          </button>
          <button className="button-danger" onClick={applyBackfill} disabled={running !== null || !report || report.applied}>
            {running === 'apply' ? 'Application…' : report?.applied ? 'Déjà appliqué' : 'Appliquer (sauvegarde automatique)'}
          </button>
        </div>
        {backfillError && <p className="form-error">{backfillError}</p>}
        {report && (
          <div className="report-grid mono">
            <div>
              <span>Lecteurs distincts</span>
              <strong>{report.distinct_users}</strong>
            </div>
            <div>
              <span>Positifs trouvés</span>
              <strong>{report.positives_found}</strong>
            </div>
            <div>
              <span>Négatifs échantillonnés</span>
              <strong>{report.negatives_sampled}</strong>
            </div>
            <div>
              <span>Vues journalisées</span>
              <strong>{report.real_views}</strong>
            </div>
            <div>
              <span>Échantillons entraînés</span>
              <strong>{report.samples_trained}</strong>
            </div>
            <div>
              <span>Taux résultant*</span>
              <strong>{pct(report.resulting_global_ctr, 2)}</strong>
            </div>
            {report.applied && (
              <div className="report-applied">
                Appliqué{report.backup_path ? ` — ancien modèle sauvegardé (${report.backup_path})` : ''}. Un redémarrage du
                service est nécessaire pour le charger.
              </div>
            )}
          </div>
        )}
        {report && (
          <p className="panel-footnote">
            *Estimation : le suivi de vue a des trous connus côté web, ce chiffre reste approximatif — les poids appris sont
            la partie fiable.
          </p>
        )}
      </Panel>

      <Panel title="Toutes les dimensions">
        <table className="dim-table mono">
          <tbody>
            {DIMENSION_ORDER.map((key) => (
              <tr key={key}>
                <td>{DIMENSION_LABELS[key].short}</td>
                <td className="dim-name">{DIMENSION_LABELS[key].name}</td>
                <td className="dim-value">{pct(stats.weights[key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  )
}
