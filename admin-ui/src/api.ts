// Client pour l'API admin de twitninf-recommender. Le service tourne en
// same-origin une fois ce build servi par `/admin/panel` (voir vite.config.ts
// — bundle single-file, embarqué dans le binaire Rust) : pas de configuration
// d'hôte nécessaire, `fetch('/admin/...')` suffit.

export interface AlgoWeights {
  d1_engagement_velocity: number
  d2_content_intelligence: number
  d3_social_graph: number
  d4_temporal: number
  d5_behavioral: number
  d6_diversity: number
  d7_viral: number
  d8_personalization: number
  d9_llm_understanding: number
}

export const DIMENSION_LABELS: Record<keyof AlgoWeights, { short: string; name: string }> = {
  d1_engagement_velocity: { short: 'D1', name: 'Engagement velocity' },
  d2_content_intelligence: { short: 'D2', name: 'Content intelligence' },
  d3_social_graph: { short: 'D3', name: 'Social graph' },
  d4_temporal: { short: 'D4', name: 'Temporal' },
  d5_behavioral: { short: 'D5', name: 'Behavioral' },
  d6_diversity: { short: 'D6', name: 'Diversity' },
  d7_viral: { short: 'D7', name: 'Viral' },
  d8_personalization: { short: 'D8', name: 'Personalization' },
  d9_llm_understanding: { short: 'D9', name: 'LLM understanding' },
}

export const DIMENSION_ORDER = Object.keys(DIMENSION_LABELS) as (keyof AlgoWeights)[]

export interface AlgoStats {
  ctr_samples: number
  global_ctr: number
  weights: AlgoWeights
  auto_tuned: boolean
  ml_active: boolean
  dwell_samples: number
  dwell_mean_weight: number
  dwell_active: boolean
  algorithm_version: string
}

export interface BackfillReport {
  since_days: number
  distinct_users: number
  positives_found: number
  negatives_sampled: number
  real_views: number
  samples_trained: number
  resulting_global_ctr: number
  resulting_weights: number[]
  applied: boolean
  backup_path: string | null
}

export interface ShadowbannedUser {
  user_id: string
  level: string
  reason: string | null
}

export interface BannedUser {
  user_id: string
  reason: string | null
}

export interface FiltersResponse {
  shadowbanned: ShadowbannedUser[]
  hard_banned: BannedUser[]
  total_shadowbanned: number
  total_hard_banned: number
}

export interface LogEntry {
  timestamp: string
  level: string
  message: string
  target: string
}

export type ShadowbanLevel = 'Clean' | 'Monitoring' | 'Suppressed' | 'Ghosted'

/** Miroir de `SetWeightsRequest` côté Rust : un champ par dimension, tous optionnels. */
export interface SetWeightsRequest {
  d1?: number
  d2?: number
  d3?: number
  d4?: number
  d5?: number
  d6?: number
  d7?: number
  d8?: number
  d9?: number
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, key: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      'X-Admin-Key': key,
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const message = body?.error || `${res.status} ${res.statusText}`
    throw new ApiError(message, res.status)
  }
  return body as T
}

export const api = {
  stats: (key: string) => request<AlgoStats>('/admin/algo/stats', key),

  weights: (key: string) =>
    request<{ weights: AlgoWeights; auto_tuned: boolean; ctr_samples: number; global_ctr: number }>(
      '/admin/algo/weights',
      key,
    ),

  setWeights: (key: string, partial: SetWeightsRequest) =>
    request<{ success: boolean; message: string; weights: AlgoWeights }>('/admin/algo/weights', key, {
      method: 'POST',
      body: JSON.stringify(partial),
    }),

  resetWeights: (key: string) =>
    request<{ success: boolean; message: string }>('/admin/algo/weights/reset', key, { method: 'POST' }),

  backfillCtr: (key: string, sinceDays: number, apply: boolean) =>
    request<{ success: boolean; report: BackfillReport }>('/admin/algo/backfill-ctr', key, {
      method: 'POST',
      body: JSON.stringify({ since_days: sinceDays, apply }),
    }),

  filters: (key: string) => request<FiltersResponse>('/admin/filters', key),

  setShadowban: (key: string, userId: string, level: ShadowbanLevel, reason: string, expiresInDays: number | null) =>
    request<{ success: boolean; message: string }>('/admin/shadowban', key, {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, level, reason: reason || null, expires_in_days: expiresInDays }),
    }),

  ban: (key: string, userId: string, reason: string) =>
    request<{ success: boolean; message: string }>('/admin/ban', key, {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, reason: reason || null }),
    }),

  unban: (key: string, userId: string) =>
    request<{ success: boolean; message: string }>('/admin/unban', key, {
      method: 'POST',
      body: JSON.stringify({ user_id: userId }),
    }),

  logs: (key: string) => request<{ logs: LogEntry[]; count: number }>('/admin/logs', key),
}
