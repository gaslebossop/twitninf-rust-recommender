import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type PrequentialHead, type TasteStatus } from '../api'
import { Panel, Pill, Stat } from '../components/Stat'

// Rafraîchi plus vite que le tableau de bord (20 s) : cette page sert à
// REGARDER un modèle qui apprend, pas à consulter un état stable.
const REFRESH_MS = 10_000
// Points gardés pour les courbes. 90 x 10 s = un quart d'heure de recul, ce qui
// suffit à voir une pente sans encombrer la mémoire de l'onglet.
const HISTORY = 90

type Point = { t: number; auc: number | null; examples: number }

function ago(ts: number | null | undefined) {
  if (!ts) return '—'
  const s = Math.max(0, Math.round(Date.now() / 1000 - ts))
  if (s < 60) return `il y a ${s} s`
  if (s < 3600) return `il y a ${Math.round(s / 60)} min`
  if (s < 86400) return `il y a ${Math.round(s / 3600)} h`
  return `il y a ${Math.round(s / 86400)} j`
}

function duration(s: number) {
  if (s < 60) return `${s} s`
  if (s < 3600) return `${Math.round(s / 60)} min`
  if (s < 86400) return `${(s / 3600).toFixed(1)} h`
  return `${(s / 86400).toFixed(1)} j`
}

/// ── Le verdict ────────────────────────────────────────────────────────────
///
/// La question posée à cette page est « est-ce que ça va ». Elle doit y
/// répondre en une phrase, avant tout chiffre. Les cas sont ordonnés du plus
/// grave au plus bénin, et surtout : « ne contribue pas » se décline en
/// plusieurs situations très différentes qu'un seul voyant confondrait.
function verdict(s: TasteStatus): { tone: 'positive' | 'negative' | 'idle' | 'accent'; title: string; detail: string } {
  const e = s.engine
  if (!s.service) {
    return {
      tone: 'negative',
      title: 'Le service ne répond pas',
      detail:
        "Le moteur ne peut plus le joindre. Le fil continue de fonctionner : il est classé exactement comme avant l'ajout du modèle. Vérifier `systemctl status taste-model` sur le VPS.",
    }
  }
  if (s.service.last_error) {
    return {
      tone: 'negative',
      title: 'Le service a rencontré une erreur',
      detail: s.service.last_error,
    }
  }
  if (!s.enabled) {
    return {
      tone: 'idle',
      title: 'Débranché du classement',
      detail:
        "Le modèle continue d'apprendre en arrière-plan, mais il ne pèse sur aucun fil. C'est l'état sûr : rien de ce qu'il fait n'atteint les lecteurs.",
    }
  }
  if (e.calls > 0 && e.failures / e.calls > 0.2) {
    return {
      tone: 'negative',
      title: 'Branché, mais il échoue souvent',
      detail: `${e.failures} échecs sur ${e.calls} appels. Le fil marche — le modèle n'y est pour rien. C'est le cas le plus trompeur : rien ne casse visiblement.`,
    }
  }
  if (!e.warm) {
    return {
      tone: 'accent',
      title: "Branché, en cours d'échauffement",
      detail:
        "Il faut 200 observations avant que l'échelle du modèle veuille dire quelque chose. En attendant il reste muet, et le classement est celui d'avant.",
    }
  }
  return {
    tone: 'positive',
    title: 'Branché et en marche',
    detail: `Il pèse ${(e.share * 100).toFixed(1)} % du score final, répond en ${e.last_latency_ms} ms, ${e.failures} échec${e.failures > 1 ? 's' : ''} sur ${e.calls} appels.`,
  }
}

/// Courbe minimale, en SVG inline. Pas de bibliothèque : une seule série, pas
/// d'interaction, et le bundle est embarqué dans le binaire Rust.
function Spark({ points, format }: { points: (number | null)[]; format: (v: number) => string }) {
  const vals = points.filter((v): v is number => v !== null)
  if (vals.length < 2) {
    return <p className="panel-note">Pas encore assez de points — la courbe se remplit pendant que la page reste ouverte.</p>
  }
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  // Plage plancher : sans elle, une série constante remplirait toute la hauteur
  // et donnerait l'illusion d'une variation énorme.
  const span = Math.max(max - min, 1e-6)
  const w = 320
  const h = 56
  const d = points
    .map((v, i) => {
      if (v === null) return null
      const x = (i / Math.max(points.length - 1, 1)) * w
      const y = h - ((v - min) / span) * (h - 8) - 4
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .filter(Boolean)
    .join(' ')
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none" role="img">
        <polyline points={d} fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.85" />
      </svg>
      <p className="panel-note mono">
        min {format(min)} · max {format(max)} · dernier {format(vals[vals.length - 1])}
      </p>
    </div>
  )
}

function HeadRow({ name, h }: { name: string; h: PrequentialHead }) {
  // 0,50 = le modèle n'ordonne rien. C'est le seul repère qui compte pour lire
  // une AUC, donc il est écrit à côté du chiffre plutôt que sous-entendu.
  const tone = h.auc >= 0.6 ? 'positive' : h.auc >= 0.52 ? 'idle' : 'negative'
  return (
    <tr>
      <td className="dim-name">{name}</td>
      <td className={`dim-value mono tone-${tone}`}>{h.auc.toFixed(3)}</td>
      <td className="dim-value mono">{h.log_loss.toFixed(3)}</td>
      <td className="dim-value mono">{h.ece.toFixed(3)}</td>
      <td className="dim-value mono">{h.n.toLocaleString('fr-FR')}</td>
    </tr>
  )
}

export function Model({ apiKey }: { apiKey: string }) {
  const [status, setStatus] = useState<TasteStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [updated, setUpdated] = useState<Date | null>(null)
  const [toggling, setToggling] = useState(false)
  const [history, setHistory] = useState<Point[]>([])
  const seen = useRef<number>(-1)

  const load = useCallback(async () => {
    try {
      const s = await api.taste(apiKey)
      setStatus(s)
      setUpdated(new Date())
      setError(null)
      // La série vit dans l'onglet : le service ne garde aucun historique, et
      // lui en faire garder un pour cette page seule serait payer un stockage
      // pour un usage qui dure le temps d'un regard.
      if (s.service) {
        const auc = s.service.prequential?.like?.auc ?? null
        const ex = s.service.examples_seen
        if (ex !== seen.current || auc !== null) {
          seen.current = ex
          setHistory((prev) => [...prev, { t: Date.now(), auc, examples: ex }].slice(-HISTORY))
        }
      }
    } catch {
      setError('Lecture impossible — le moteur ne répond pas.')
    }
  }, [apiKey])

  useEffect(() => {
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => clearInterval(id)
  }, [load])

  async function toggle() {
    if (!status) return
    setToggling(true)
    try {
      await api.setTaste(apiKey, !status.enabled)
      await load()
    } catch {
      setError("La bascule a échoué — l'état affiché n'a pas changé.")
    } finally {
      setToggling(false)
    }
  }

  if (error && !status) return <div className="empty-state">{error}</div>
  if (!status) return <div className="loading">Lecture…</div>

  const v = verdict(status)
  const e = status.engine
  const s = status.service
  const heads = s?.prequential ? Object.entries(s.prequential) : []
  const tables = s?.sparse_tables ? Object.entries(s.sparse_tables).filter(([, t]) => t.rows > 0) : []
  const totalRows = tables.reduce((a, [, t]) => a + t.rows, 0)

  return (
    <div className="dashboard">
      <div className="hero">
        <div className="hero-status">
          <Pill tone={v.tone}>{v.title}</Pill>
          <Pill tone={status.enabled ? 'accent' : 'idle'}>
            {status.enabled ? `dans le fil · ${(e.share * 100).toFixed(1)} % du score` : 'hors du fil'}
          </Pill>
          {s && <Pill tone="idle">en service depuis {duration(s.uptime_s)}</Pill>}
        </div>
        <h1>Modèle de goûts — taste-model</h1>
        <p className="panel-note">{v.detail}</p>
        {updated && (
          <p className="hero-meta mono">
            actualisé il y a {Math.max(0, Math.round((Date.now() - updated.getTime()) / 1000))} s · rafraîchi toutes les{' '}
            {REFRESH_MS / 1000} s
          </p>
        )}
      </div>

      <Panel
        title="Interrupteur"
        action={
          <button
            className={status.enabled ? 'button-danger' : 'button-primary'}
            onClick={toggle}
            disabled={toggling}
          >
            {toggling ? '…' : status.enabled ? 'Débrancher du fil' : 'Brancher sur le fil'}
          </button>
        }
      >
        <p className="panel-note">
          {status.enabled
            ? "Le modèle pèse sur le classement de tous les lecteurs. Le débrancher prend effet à la requête de fil suivante — il n'y a rien à redémarrer, et le service continue d'apprendre pendant ce temps."
            : "Le modèle est retiré du classement. Il continue d'apprendre : le rebrancher ne repart pas de zéro."}
        </p>
      </Panel>

      <div className="grid-3">
        <Panel title="Contribution au fil">
          <div className="stat-row">
            <Stat label="Appels" value={e.calls.toLocaleString('fr-FR')} />
            <Stat
              label="Échecs"
              value={e.failures.toLocaleString('fr-FR')}
              tone={e.failures === 0 ? 'positive' : 'negative'}
            />
            <Stat
              label="Dépassements"
              value={e.timeouts.toLocaleString('fr-FR')}
              tone={e.timeouts === 0 ? 'positive' : 'negative'}
            />
          </div>
          <div className="stat-row">
            <Stat label="Dernière latence" value={e.last_latency_ms} unit="ms" />
            <Stat label="Probabilité moyenne" value={e.mean_p.toFixed(3)} hint="sert d'échelle au mélange" />
          </div>
          <div className="stat-row">
            <Stat
              label="Part du score final"
              value={`${(e.share * 100).toFixed(1)} %`}
              hint={`poids relatif ${e.weight.toFixed(2)}`}
            />
          </div>
          <p className="panel-note">
            La <strong>part</strong> n'est pas le poids : le mélange est une moyenne pondérée renormalisée sur les
            termes disponibles, donc un poids de {e.weight.toFixed(2)} pèse d'autant plus que les autres têtes sont
            froides. C'est la part qu'il faut lire, jamais le poids seul.
          </p>
          <p className="panel-note">
            Un échec n'abîme rien : le moteur classe alors exactement comme avant l'ajout du modèle. Ce qu'il faut
            surveiller, c'est un nombre d'échecs qui suit celui des appels — le fil marcherait, sans que le modèle y soit
            pour quoi que ce soit.
          </p>
        </Panel>

        <Panel title="Entraînement continu">
          {s ? (
            <>
              <div className="stat-row">
                <Stat label="Tours" value={s.train_rounds.toLocaleString('fr-FR')} />
                <Stat label="Exemples digérés" value={s.examples_seen.toLocaleString('fr-FR')} />
              </div>
              <div className="stat-row">
                <Stat label="Dernier tour" value={ago(s.last_train)} />
                <Stat label="Sauvegardes" value={`${s.sparse_saves} / ${s.dense_saves}`} hint="creuses / denses" />
              </div>
              <p className="panel-note">
                Un tour qui ne consomme rien est normal : s'il n'est arrivé aucune impression nouvelle, il n'y a rien à
                apprendre et les poids ne bougent pas. Le modèle ne se dégrade pas quand l'app est calme, il attend.
              </p>
            </>
          ) : (
            <p className="panel-note">Service injoignable.</p>
          )}
        </Panel>

        <Panel title="Taille du modèle">
          {s ? (
            <>
              <div className="stat-row">
                <Stat label="Denses" value={s.params_dense.toLocaleString('fr-FR')} hint="figés à l'architecture" />
                <Stat
                  label="Creux"
                  value={s.params_sparse.toLocaleString('fr-FR')}
                  hint="grandissent avec le corpus"
                />
              </div>
              <div className="stat-row">
                <Stat label="Identifiants admis" value={totalRows.toLocaleString('fr-FR')} />
                <Stat label="Tweets au catalogue" value={s.catalog_size.toLocaleString('fr-FR')} />
              </div>
              <p className="panel-note">
                Les paramètres creux ne sont pas une taille choisie : c'est ce que le corpus a mérité. Chaque identifiant
                doit avoir été vu assez de fois pour obtenir sa ligne.
              </p>
            </>
          ) : (
            <p className="panel-note">Service injoignable.</p>
          )}
        </Panel>
      </div>

      <div className="grid-2">
        <Panel title="Qualité mesurée — validation progressive">
          {heads.length ? (
            <table className="dim-table">
              <thead>
                <tr>
                  <th>Tête</th>
                  <th>AUC</th>
                  <th>Log-loss</th>
                  <th>ECE</th>
                  <th>n</th>
                </tr>
              </thead>
              <tbody>
                {heads.map(([name, h]) => (
                  <HeadRow key={name} name={name} h={h} />
                ))}
              </tbody>
            </table>
          ) : (
            <p className="panel-note">
              Pas encore 50 observations par tête. Ce tableau reste vide tant qu'un chiffre inviterait à conclure sur
              trop peu de monde.
            </p>
          )}
          <p className="panel-note">
            Chaque lot est <strong>prédit avant d'être appris</strong>. C'est ce qui rend ces chiffres honnêtes : un
            modèle qui apprend en continu et qu'on noterait après coup réciterait au lieu de prédire, et son AUC
            monterait toute seule sans rien vouloir dire.
          </p>
          <p className="panel-footnote">
            AUC à 0,50 = le modèle n'ordonne rien, il décale tous les scores pareil. C'est le chiffre à regarder en
            premier.
          </p>
        </Panel>

        <Panel title="Évolution depuis l'ouverture de la page">
          <p className="stat-label">AUC de la tête « like »</p>
          <Spark points={history.map((p) => p.auc)} format={(x) => x.toFixed(3)} />
          <p className="stat-label" style={{ marginTop: 16 }}>
            Exemples digérés
          </p>
          <Spark points={history.map((p) => p.examples)} format={(x) => x.toLocaleString('fr-FR')} />
          <p className="panel-footnote">
            Série tenue dans cet onglet, pas côté serveur : elle repart à zéro au rechargement. Pour un historique
            durable, c'est `/stats` du service qu'il faudrait enregistrer.
          </p>
        </Panel>
      </div>

      {tables.length > 0 && (
        <Panel title="Tables d'embedding sans collision">
          <table className="dim-table">
            <thead>
              <tr>
                <th>Fente</th>
                <th>Lignes</th>
                <th>Dim.</th>
                <th>Paramètres</th>
              </tr>
            </thead>
            <tbody>
              {tables
                .sort((a, b) => b[1].rows * b[1].dim - a[1].rows * a[1].dim)
                .map(([name, t]) => (
                  <tr key={name}>
                    <td className="dim-name">{name}</td>
                    <td className="dim-value mono">{t.rows.toLocaleString('fr-FR')}</td>
                    <td className="dim-value mono">{t.dim}</td>
                    <td className="dim-value mono">{(t.rows * t.dim).toLocaleString('fr-FR')}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="panel-note">
            Les fentes en <code>x_</code> sont des croisements — « ce lecteur-là aime CE thème-là ». C'est là que vit la
            personnalisation, et c'est aussi la fente qui grandit le plus vite.
          </p>
        </Panel>
      )}

      {error && <p className="form-error">{error}</p>}
    </div>
  )
}
