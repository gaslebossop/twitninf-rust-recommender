import { DIMENSION_LABELS, DIMENSION_ORDER, type AlgoWeights } from '../api'

/**
 * Lecture façon égaliseur : neuf bandes, une par dimension, hauteur = poids.
 * C'est la seule vue qui répond directement à « qu'est-ce que l'auto-tuner a
 * appris » — un tableau de nombres ne se lit pas d'un coup d'œil, neuf barres
 * si. `compare` superpose une seconde série (fantôme) pour juger un
 * rattrapage avant de l'appliquer.
 */
export function WeightBars({
  weights,
  compare,
  compareLabel,
}: {
  weights: AlgoWeights
  compare?: AlgoWeights
  compareLabel?: string
}) {
  const max = Math.max(
    0.05,
    ...DIMENSION_ORDER.map((k) => weights[k]),
    ...(compare ? DIMENSION_ORDER.map((k) => compare[k]) : []),
  )

  return (
    <div className="eq">
      {DIMENSION_ORDER.map((key) => {
        const value = weights[key]
        const compareValue = compare?.[key]
        const pct = Math.max(2, (value / max) * 100)
        const comparePct = compareValue !== undefined ? Math.max(2, (compareValue / max) * 100) : null
        return (
          <div className="eq-col" key={key}>
            <div className="eq-track">
              {comparePct !== null && <div className="eq-ghost" style={{ height: `${comparePct}%` }} />}
              <div className="eq-bar" style={{ height: `${pct}%` }} />
              <span className="eq-value mono">{(value * 100).toFixed(1)}</span>
            </div>
            <span className="eq-label mono">{DIMENSION_LABELS[key].short}</span>
          </div>
        )
      })}
      {compare && (
        <div className="eq-legend">
          <span>
            <i className="eq-swatch eq-swatch-live" /> actif
          </span>
          <span>
            <i className="eq-swatch eq-swatch-ghost" /> {compareLabel ?? 'comparaison'}
          </span>
        </div>
      )}
    </div>
  )
}
