use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::Result;
use deadpool_postgres::Pool as PgPool;
use rand::Rng;
use serde::Serialize;
use tracing::{debug, info};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentAssignment {
    pub experiment_id: String,
    pub tweet_id: String,
    pub variant_id: String,
    pub variant_label: String,
    pub content: String,
    pub status: String,
    pub is_winner: bool,
}

#[derive(Debug, Clone)]
struct VariantPerformance {
    id: String,
    label: String,
    content: String,
    position: i32,
    impressions: i64,
    reward: f64,
}

#[derive(Debug, Clone)]
struct ExperimentSnapshot {
    id: String,
    tweet_id: String,
    status: String,
    winner_variant_id: Option<String>,
    exploration_percent: i32,
    min_impressions_per_variant: i32,
    variants: Vec<VariantPerformance>,
}

#[derive(Debug, Clone)]
pub struct WinnerUpdate {
    pub experiment_id: String,
    pub variant_id: String,
}

/// Crée uniquement les structures manquantes. La même définition vit dans la
/// migration Node afin que les deux services puissent être déployés séparément.
pub async fn ensure_schema(pg: &PgPool) -> Result<()> {
    let client = pg.get().await?;
    client
        .batch_execute(
            r#"
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

        CREATE TABLE IF NOT EXISTS tweet_ab_experiments (
            id UUID PRIMARY KEY,
            tweet_id UUID NOT NULL UNIQUE REFERENCES tweets(id) ON DELETE CASCADE,
            author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            status VARCHAR(20) NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'active', 'completed', 'cancelled')),
            strategy VARCHAR(20) NOT NULL DEFAULT 'adaptive',
            platform_scope VARCHAR(20) NOT NULL DEFAULT 'windows',
            exploration_percent SMALLINT NOT NULL DEFAULT 20
                CHECK (exploration_percent BETWEEN 0 AND 100),
            min_impressions_per_variant INTEGER NOT NULL DEFAULT 6
                CHECK (min_impressions_per_variant >= 4),
            winner_variant_id UUID NULL,
            cancellation_reason TEXT NULL,
            activated_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS tweet_ab_variants (
            id UUID PRIMARY KEY,
            experiment_id UUID NOT NULL REFERENCES tweet_ab_experiments(id) ON DELETE CASCADE,
            position SMALLINT NOT NULL,
            label VARCHAR(4) NOT NULL,
            content TEXT NOT NULL,
            is_control BOOLEAN NOT NULL DEFAULT FALSE,
            moderation_status VARCHAR(20) NOT NULL DEFAULT 'pending'
                CHECK (moderation_status IN ('pending', 'approved', 'rejected')),
            moderation_reason TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (experiment_id, position),
            UNIQUE (experiment_id, label)
        );

        CREATE TABLE IF NOT EXISTS tweet_ab_variant_metrics (
            variant_id UUID PRIMARY KEY REFERENCES tweet_ab_variants(id) ON DELETE CASCADE,
            impressions BIGINT NOT NULL DEFAULT 0,
            interactions BIGINT NOT NULL DEFAULT 0,
            reward DOUBLE PRECISION NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS tweet_ab_assignments (
            experiment_id UUID NOT NULL REFERENCES tweet_ab_experiments(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            variant_id UUID NOT NULL REFERENCES tweet_ab_variants(id) ON DELETE CASCADE,
            assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (experiment_id, user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_tweet_ab_experiments_status
            ON tweet_ab_experiments (status, tweet_id);
        CREATE INDEX IF NOT EXISTS idx_tweet_ab_variants_experiment
            ON tweet_ab_variants (experiment_id, moderation_status);
        CREATE INDEX IF NOT EXISTS idx_tweet_ab_assignments_user
            ON tweet_ab_assignments (user_id, experiment_id);

        ALTER TABLE tweet_ab_experiments
            DROP CONSTRAINT IF EXISTS tweet_ab_experiments_min_impressions_per_variant_check;
        ALTER TABLE tweet_ab_experiments
            ALTER COLUMN min_impressions_per_variant SET DEFAULT 6;
        ALTER TABLE tweet_ab_experiments
            ADD CONSTRAINT tweet_ab_experiments_min_impressions_per_variant_check
            CHECK (min_impressions_per_variant >= 4);

        WITH variant_counts AS (
            SELECT experiment_id, COUNT(*)::int AS variant_count
            FROM tweet_ab_variants
            GROUP BY experiment_id
        )
        UPDATE tweet_ab_experiments e
        SET min_impressions_per_variant =
                GREATEST(4, CEIL(16.0 / vc.variant_count)::int),
            updated_at = NOW()
        FROM variant_counts vc
        WHERE e.id = vc.experiment_id
          AND e.status IN ('pending', 'active')
          AND vc.variant_count >= 2;
    "#,
        )
        .await?;
    Ok(())
}

fn smoothed_reward(variant: &VariantPerformance) -> f64 {
    // Petit a priori neutre : évite qu'un unique like fasse gagner une version
    // qui n'a encore reçu presque aucune impression.
    (variant.reward + 0.5) / (variant.impressions.max(0) as f64 + 5.0)
}

fn best_variant(variants: &[VariantPerformance]) -> Option<&VariantPerformance> {
    variants.iter().max_by(|left, right| {
        smoothed_reward(left)
            .partial_cmp(&smoothed_reward(right))
            .unwrap_or(Ordering::Equal)
            // À score égal, conserver la version la plus proche du contrôle.
            .then_with(|| right.position.cmp(&left.position))
    })
}

fn choose_active_variant(experiment: &ExperimentSnapshot) -> Option<&VariantPerformance> {
    if experiment.variants.is_empty() {
        return None;
    }

    let mut rng = rand::thread_rng();
    // Avec une quarantaine d'utilisateurs actifs, chaque lecture compte :
    // remplir d'abord équitablement le petit budget de chaque variante évite
    // qu'une version soit affamée par une exploitation trop précoce.
    let minimum = i64::from(experiment.min_impressions_per_variant.max(4));
    let lowest_impressions = experiment
        .variants
        .iter()
        .map(|variant| variant.impressions)
        .min()
        .unwrap_or(0);
    if lowest_impressions < minimum {
        let least_exposed: Vec<&VariantPerformance> = experiment
            .variants
            .iter()
            .filter(|variant| variant.impressions == lowest_impressions)
            .collect();
        return least_exposed
            .get(rng.gen_range(0..least_exposed.len()))
            .copied();
    }

    if rng.gen_range(0..100) < experiment.exploration_percent.clamp(0, 100) {
        let index = rng.gen_range(0..experiment.variants.len());
        experiment.variants.get(index)
    } else {
        best_variant(&experiment.variants)
    }
}

fn winner_candidate(
    variants: &[VariantPerformance],
    min_impressions_per_variant: i32,
) -> Option<&VariantPerformance> {
    if variants.len() < 2 {
        return None;
    }

    let minimum = i64::from(min_impressions_per_variant.max(4));
    if variants.iter().any(|variant| variant.impressions < minimum) {
        return None;
    }

    let best = best_variant(variants)?;
    let mut runner_up_score = f64::NEG_INFINITY;
    for variant in variants {
        if variant.id == best.id {
            continue;
        }
        runner_up_score = runner_up_score.max(smoothed_reward(variant));
    }

    let lift = smoothed_reward(best) - runner_up_score;
    let reached_hard_stop = variants
        .iter()
        .all(|variant| variant.impressions >= minimum.saturating_mul(2));

    if lift >= 0.02 || reached_hard_stop {
        Some(best)
    } else {
        None
    }
}

/// Retourne une variante stable pour chaque tweet expérimental de la page.
/// Les lecteurs déjà affectés gardent leur version tant que le test est actif.
pub async fn assign_variants(
    pg: &PgPool,
    user_id: &str,
    tweet_ids: &[String],
) -> Result<Vec<ExperimentAssignment>> {
    let user_uuid = Uuid::parse_str(user_id)?;
    let tweet_uuids: Vec<Uuid> = tweet_ids
        .iter()
        .filter_map(|id| Uuid::parse_str(id).ok())
        .collect();
    if tweet_uuids.is_empty() {
        return Ok(Vec::new());
    }

    let client = pg.get().await?;
    let rows = client
        .query(
            r#"
        SELECT
            e.id::text,
            e.tweet_id::text,
            e.status,
            e.winner_variant_id::text,
            e.exploration_percent::int,
            e.min_impressions_per_variant::int,
            v.id::text,
            v.label,
            v.content,
            v.position::int,
            COALESCE(m.impressions, 0)::bigint,
            COALESCE(m.reward, 0)::float8
        FROM tweet_ab_experiments e
        JOIN tweet_ab_variants v
          ON v.experiment_id = e.id
         AND v.moderation_status = 'approved'
        LEFT JOIN tweet_ab_variant_metrics m ON m.variant_id = v.id
        WHERE e.tweet_id = ANY($1)
          AND e.status IN ('active', 'completed')
        ORDER BY e.tweet_id, v.position
    "#,
            &[&tweet_uuids],
        )
        .await?;

    let mut experiments: HashMap<String, ExperimentSnapshot> = HashMap::new();
    for row in rows {
        let experiment_id: String = row.get(0);
        let entry = experiments
            .entry(row.get::<_, String>(1))
            .or_insert_with(|| ExperimentSnapshot {
                id: experiment_id,
                tweet_id: row.get(1),
                status: row.get(2),
                winner_variant_id: row.get(3),
                exploration_percent: row.get(4),
                min_impressions_per_variant: row.get(5),
                variants: Vec::new(),
            });
        entry.variants.push(VariantPerformance {
            id: row.get(6),
            label: row.get(7),
            content: row.get(8),
            position: row.get(9),
            impressions: row.get(10),
            reward: row.get(11),
        });
    }
    if experiments.is_empty() {
        return Ok(Vec::new());
    }

    let experiment_uuids: Vec<Uuid> = experiments
        .values()
        .filter_map(|experiment| Uuid::parse_str(&experiment.id).ok())
        .collect();
    let assignment_rows = client
        .query(
            r#"
        SELECT experiment_id::text, variant_id::text
        FROM tweet_ab_assignments
        WHERE user_id = $1 AND experiment_id = ANY($2)
    "#,
            &[&user_uuid, &experiment_uuids],
        )
        .await?;
    let existing: HashMap<String, String> = assignment_rows
        .into_iter()
        .map(|row| (row.get(0), row.get(1)))
        .collect();

    let mut assignments = Vec::new();
    for tweet_id in tweet_ids {
        let Some(experiment) = experiments.get(tweet_id) else {
            continue;
        };

        let chosen = if experiment.status == "completed" {
            experiment
                .winner_variant_id
                .as_ref()
                .and_then(|winner| {
                    experiment
                        .variants
                        .iter()
                        .find(|variant| &variant.id == winner)
                })
                .or_else(|| best_variant(&experiment.variants))
        } else {
            existing
                .get(&experiment.id)
                .and_then(|variant_id| {
                    experiment
                        .variants
                        .iter()
                        .find(|variant| &variant.id == variant_id)
                })
                .or_else(|| choose_active_variant(experiment))
        };
        let Some(chosen) = chosen else { continue };
        let experiment_uuid = Uuid::parse_str(&experiment.id)?;
        let variant_uuid = Uuid::parse_str(&chosen.id)?;

        if experiment.status == "completed" {
            // Une fois le gagnant choisi, les affectations deviennent toutes le
            // gagnant. Les métriques historiques sont séparées et restent intactes.
            client
                .execute(
                    r#"
                INSERT INTO tweet_ab_assignments (
                    experiment_id, user_id, variant_id, assigned_at, last_seen_at
                ) VALUES ($1, $2, $3, NOW(), NOW())
                ON CONFLICT (experiment_id, user_id) DO UPDATE
                SET variant_id = EXCLUDED.variant_id, last_seen_at = NOW()
            "#,
                    &[&experiment_uuid, &user_uuid, &variant_uuid],
                )
                .await?;
        } else {
            let inserted = client
                .execute(
                    r#"
                INSERT INTO tweet_ab_assignments (
                    experiment_id, user_id, variant_id, assigned_at, last_seen_at
                ) VALUES ($1, $2, $3, NOW(), NOW())
                ON CONFLICT (experiment_id, user_id) DO NOTHING
            "#,
                    &[&experiment_uuid, &user_uuid, &variant_uuid],
                )
                .await?;

            if inserted == 0 {
                client
                    .execute(
                        r#"
                    UPDATE tweet_ab_assignments
                    SET last_seen_at = NOW()
                    WHERE experiment_id = $1 AND user_id = $2
                "#,
                        &[&experiment_uuid, &user_uuid],
                    )
                    .await?;
            }
        }

        // Une insertion concurrente peut avoir gagné le conflit : relire la
        // vérité stockée avant de rendre le contenu au lecteur.
        let stored_variant: String = client
            .query_one(
                r#"
            SELECT variant_id::text
            FROM tweet_ab_assignments
            WHERE experiment_id = $1 AND user_id = $2
        "#,
                &[&experiment_uuid, &user_uuid],
            )
            .await?
            .get(0);
        let Some(stored) = experiment
            .variants
            .iter()
            .find(|variant| variant.id == stored_variant)
        else {
            continue;
        };

        assignments.push(ExperimentAssignment {
            experiment_id: experiment.id.clone(),
            tweet_id: experiment.tweet_id.clone(),
            variant_id: stored.id.clone(),
            variant_label: stored.label.clone(),
            content: stored.content.clone(),
            status: experiment.status.clone(),
            is_winner: experiment.winner_variant_id.as_deref() == Some(stored.id.as_str()),
        });
    }

    debug!(
        user_id,
        assignments = assignments.len(),
        "A/B variants assigned"
    );
    Ok(assignments)
}

async fn maybe_finalize(
    pg: &PgPool,
    experiment_id: &str,
    min_impressions_per_variant: i32,
) -> Result<Option<WinnerUpdate>> {
    let experiment_uuid = Uuid::parse_str(experiment_id)?;
    let client = pg.get().await?;
    let rows = client
        .query(
            r#"
        SELECT
            v.id::text,
            v.label,
            v.content,
            v.position::int,
            COALESCE(m.impressions, 0)::bigint,
            COALESCE(m.reward, 0)::float8
        FROM tweet_ab_variants v
        LEFT JOIN tweet_ab_variant_metrics m ON m.variant_id = v.id
        WHERE v.experiment_id = $1 AND v.moderation_status = 'approved'
        ORDER BY v.position
    "#,
            &[&experiment_uuid],
        )
        .await?;
    let variants: Vec<VariantPerformance> = rows
        .into_iter()
        .map(|row| VariantPerformance {
            id: row.get(0),
            label: row.get(1),
            content: row.get(2),
            position: row.get(3),
            impressions: row.get(4),
            reward: row.get(5),
        })
        .collect();

    let Some(winner) = winner_candidate(&variants, min_impressions_per_variant) else {
        return Ok(None);
    };
    let winner_uuid = Uuid::parse_str(&winner.id)?;
    // La promotion est atomique : le test est marqué terminé ET le tweet
    // canonique adopte le texte gagnant dans la même instruction. Les anciens
    // clients et le profil de l'auteur gardent ainsi eux aussi la meilleure
    // version après la fin de l'expérience.
    let changed = client
        .execute(
            r#"
        WITH completed AS (
            UPDATE tweet_ab_experiments
            SET status = 'completed', winner_variant_id = $2,
                completed_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND status = 'active'
            RETURNING tweet_id
        )
        UPDATE tweets t
        SET content = v.content,
            updated_at = NOW(),
            metadata = COALESCE(t.metadata, '{}'::jsonb) || jsonb_build_object(
                'ab_test',
                jsonb_build_object(
                    'status', 'completed',
                    'experiment_id', $1::text,
                    'winner_variant_id', $2::text,
                    'completed_at', NOW()
                )
            )
        FROM completed c
        JOIN tweet_ab_variants v ON v.id = $2
        WHERE t.id = c.tweet_id
    "#,
            &[&experiment_uuid, &winner_uuid],
        )
        .await?;

    if changed == 0 {
        return Ok(None);
    }

    info!(
        experiment_id,
        winner_variant_id = %winner.id,
        winner_label = %winner.label,
        score = smoothed_reward(winner),
        "A/B experiment completed"
    );
    Ok(Some(WinnerUpdate {
        experiment_id: experiment_id.to_string(),
        variant_id: winner.id.clone(),
    }))
}

/// Attribue l'interaction à la version réellement vue, puis tente de conclure
/// l'expérience. Les erreurs restent non fatales pour le tracking NeuralRank.
pub async fn record_interaction(
    pg: &PgPool,
    user_id: &str,
    tweet_id: &str,
    variant_hint: Option<&str>,
    reward: f64,
    is_impression: bool,
) -> Result<Option<WinnerUpdate>> {
    let user_uuid = Uuid::parse_str(user_id)?;
    let tweet_uuid = Uuid::parse_str(tweet_id)?;
    let client = pg.get().await?;
    let Some(row) = client
        .query_opt(
            r#"
        SELECT
            e.id::text,
            e.status,
            e.winner_variant_id::text,
            e.min_impressions_per_variant,
            a.variant_id::text
        FROM tweet_ab_experiments e
        LEFT JOIN tweet_ab_assignments a
          ON a.experiment_id = e.id AND a.user_id = $1
        WHERE e.tweet_id = $2 AND e.status IN ('active', 'completed')
    "#,
            &[&user_uuid, &tweet_uuid],
        )
        .await?
    else {
        return Ok(None);
    };

    let experiment_id: String = row.get(0);
    let status: String = row.get(1);
    let winner_variant_id: Option<String> = row.get(2);
    let min_impressions: i32 = row.get(3);
    let assigned_variant_id: Option<String> = row.get(4);
    let experiment_uuid = Uuid::parse_str(&experiment_id)?;

    let hinted = variant_hint
        .and_then(|value| Uuid::parse_str(value).ok())
        .map(|value| value.to_string());
    let candidate = hinted.or_else(|| {
        if status == "completed" {
            winner_variant_id.clone()
        } else {
            assigned_variant_id
        }
    });
    let Some(candidate) = candidate else {
        return Ok(None);
    };
    let variant_uuid = Uuid::parse_str(&candidate)?;

    let belongs = client
        .query_opt(
            r#"
        SELECT 1
        FROM tweet_ab_variants
        WHERE id = $1 AND experiment_id = $2 AND moderation_status = 'approved'
    "#,
            &[&variant_uuid, &experiment_uuid],
        )
        .await?
        .is_some();
    if !belongs {
        return Ok(None);
    }

    client
        .execute(
            r#"
        INSERT INTO tweet_ab_assignments (
            experiment_id, user_id, variant_id, assigned_at, last_seen_at
        ) VALUES ($1, $2, $3, NOW(), NOW())
        ON CONFLICT (experiment_id, user_id) DO UPDATE
        SET variant_id = EXCLUDED.variant_id, last_seen_at = NOW()
    "#,
            &[&experiment_uuid, &user_uuid, &variant_uuid],
        )
        .await?;

    if is_impression {
        client
            .execute(
                r#"
            INSERT INTO tweet_ab_variant_metrics (
                variant_id, impressions, interactions, reward, updated_at
            ) VALUES ($1, 1, 0, 0, NOW())
            ON CONFLICT (variant_id) DO UPDATE
            SET impressions = tweet_ab_variant_metrics.impressions + 1,
                updated_at = NOW()
        "#,
                &[&variant_uuid],
            )
            .await?;
    } else {
        client
            .execute(
                r#"
            INSERT INTO tweet_ab_variant_metrics (
                variant_id, impressions, interactions, reward, updated_at
            ) VALUES ($1, 0, 1, $2, NOW())
            ON CONFLICT (variant_id) DO UPDATE
            SET interactions = tweet_ab_variant_metrics.interactions + 1,
                reward = tweet_ab_variant_metrics.reward + EXCLUDED.reward,
                updated_at = NOW()
        "#,
                &[&variant_uuid, &reward],
            )
            .await?;
    }

    if status == "active" {
        maybe_finalize(pg, &experiment_id, min_impressions).await
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn performance(id: &str, position: i32, impressions: i64, reward: f64) -> VariantPerformance {
        VariantPerformance {
            id: id.into(),
            label: String::from(char::from_u32(65 + position as u32).unwrap()),
            content: id.into(),
            position,
            impressions,
            reward,
        }
    }

    #[test]
    fn does_not_choose_a_winner_before_minimum_exposure() {
        let variants = vec![performance("a", 0, 7, 3.0), performance("b", 1, 8, 1.0)];
        assert!(winner_candidate(&variants, 8).is_none());
    }

    #[test]
    fn promotes_the_best_version_after_enough_evidence() {
        let variants = vec![performance("a", 0, 8, 1.0), performance("b", 1, 8, 4.0)];
        assert_eq!(
            winner_candidate(&variants, 8).map(|variant| variant.id.as_str()),
            Some("b")
        );
    }

    #[test]
    fn hard_stop_keeps_a_best_version_even_when_lift_is_small() {
        let variants = vec![performance("a", 0, 16, 2.0), performance("b", 1, 16, 2.1)];
        assert_eq!(
            winner_candidate(&variants, 8).map(|variant| variant.id.as_str()),
            Some("b")
        );
    }
}
