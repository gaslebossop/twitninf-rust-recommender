//! Persistance Redis du registre d'avertissements.
//!
//! Clés :
//! ```text
//! shadowban:strikes:{user_id}  → ZSET  membre = Strike en JSON, score = émission (epoch s)
//! shadowban:level:{user_id}    → cache dérivé du registre, TTL = date de retour du compte
//! admin:shadowban:{user_id}    → décision manuelle (existant), TTL facultatif désormais
//! ```
//!
//! Deux chemins de lecture, volontairement asymétriques :
//!
//! - **Chaud** (`shadowban_load_levels`) — appelé sur chaque pool de candidats,
//!   pour tous les auteurs à la fois. Deux `MGET`, rien de plus. Il lit le
//!   niveau *dérivé*, jamais le registre.
//! - **Froid** (`shadowban_load_ledger`) — un seul compte, sur ajout
//!   d'avertissement ou consultation de l'état. Lit le registre complet.
//!
//! Le cache dérivé porte un TTL calé sur l'expiration du prochain avertissement
//! ([`StrikeLedger::cache_ttl_secs`]). C'est ce qui fait tenir l'expiration à 90
//! jours **sans aucune tâche périodique** : la clé meurt d'elle-même au moment
//! exact où le niveau devrait baisser, et la lecture suivante repart du registre.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use redis::AsyncCommands;
use tracing::{debug, error, warn};

use crate::services::cache_manager::CacheManager;

use super::models::{strike_ttl, ShadowbanLevel, Strike, StrikePolicy};
use super::strikes::StrikeLedger;

fn strikes_key(user_id: &str) -> String {
    format!("shadowban:strikes:{user_id}")
}
fn level_key(user_id: &str) -> String {
    format!("shadowban:level:{user_id}")
}
fn manual_key(user_id: &str) -> String {
    format!("admin:shadowban:{user_id}")
}

/// Marqueur écrit quand le registre donne `Clean`.
///
/// Sans lui, un compte sain provoquerait un rechargement complet du registre à
/// chaque pool de candidats — c'est-à-dire dans l'immense majorité des cas.
const CLEAN_MARKER: &str = "clean";

impl CacheManager {
    // ─── Écriture ─────────────────────────────────────────────────────────────

    /// Ajoute un avertissement daté et renvoie le registre à jour.
    ///
    /// L'avertissement n'a pas besoin d'être « levé » : il s'efface seul au bout
    /// de [`STRIKE_TTL_DAYS`](super::models::STRIKE_TTL_DAYS) jours.
    pub async fn shadowban_add_strike(
        &self,
        user_id: &str,
        policy: StrikePolicy,
        tweet_id: Option<&str>,
        reason: Option<&str>,
    ) -> StrikeLedger {
        let now = Utc::now();
        let strike = Strike {
            policy,
            issued_at: now,
            tweet_id: tweet_id.map(str::to_string),
            reason: reason.map(str::to_string),
        };
        let Ok(member) = serde_json::to_string(&strike) else {
            error!(user_id, "Sérialisation d'un avertissement impossible");
            return self.shadowban_load_ledger(user_id).await;
        };

        let key = strikes_key(user_id);
        {
            let mut c = self.conn.lock().await;
            let _: Result<(), _> = c.zadd(&key, member, now.timestamp()).await;
            // La clé entière ne survit pas au dernier avertissement qu'elle porte.
            let _: Result<(), _> = c.expire(&key, strike_ttl().num_seconds()).await;
        }

        let ledger = self.shadowban_load_ledger(user_id).await;
        let level = ledger.level(now);
        self.shadowban_write_level_cache(user_id, &ledger, now)
            .await;

        if let Some(p) = ledger.permanent_ban_reached(now) {
            // Signalé, jamais appliqué automatiquement : un bannissement définitif
            // est irréversible et se déciderait ici sur des heuristiques.
            warn!(
                user_id,
                policy = p.label(),
                strikes = ledger.count_for(p, now),
                "Seuil de bannissement définitif atteint — décision humaine requise (/admin/ban)"
            );
        }

        debug!(
            user_id,
            policy = policy.label(),
            level = level.label(),
            active = ledger.active_count(now),
            "Avertissement enregistré"
        );
        ledger
    }

    /// Annule tous les avertissements d'un compte (recours accepté, erreur de
    /// détection). Le niveau redescend immédiatement.
    pub async fn shadowban_clear_strikes(&self, user_id: &str) {
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.del(strikes_key(user_id)).await;
        let _: Result<(), _> = c.del(level_key(user_id)).await;
        drop(c);
        debug!(user_id, "Registre d'avertissements vidé");
    }

    /// Annule les avertissements liés à un tweet précis.
    ///
    /// C'est la forme que prend un recours accepté chez TikTok : le contenu
    /// redevient éligible **et** l'avertissement associé disparaît du compte. Sans
    /// ce second effet, gagner son recours ne répare que la moitié du dommage.
    pub async fn shadowban_revoke_strikes_for_tweet(&self, user_id: &str, tweet_id: &str) -> u32 {
        let key = strikes_key(user_id);
        let members: Vec<String> = {
            let mut c = self.conn.lock().await;
            c.zrange(&key, 0, -1).await.unwrap_or_default()
        };

        let doomed: Vec<String> = members
            .into_iter()
            .filter(|m| {
                serde_json::from_str::<Strike>(m)
                    .ok()
                    .and_then(|s| s.tweet_id)
                    .is_some_and(|t| t == tweet_id)
            })
            .collect();

        if doomed.is_empty() {
            return 0;
        }
        let removed = doomed.len() as u32;
        {
            let mut c = self.conn.lock().await;
            for m in doomed {
                let _: Result<(), _> = c.zrem(&key, m).await;
            }
            let _: Result<(), _> = c.del(level_key(user_id)).await;
        }
        debug!(
            user_id,
            tweet_id, removed, "Avertissements annulés après recours"
        );
        removed
    }

    /// Réécrit le cache de niveau dérivé, avec un TTL calé sur la date de retour.
    async fn shadowban_write_level_cache(
        &self,
        user_id: &str,
        ledger: &StrikeLedger,
        now: DateTime<Utc>,
    ) {
        let level = ledger.computed_level(now);
        let ttl = ledger.cache_ttl_secs(now);
        let value = if level == ShadowbanLevel::Clean {
            CLEAN_MARKER
        } else {
            level.label()
        };
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.set_ex(level_key(user_id), value, ttl).await;
    }

    // ─── Lecture froide ───────────────────────────────────────────────────────

    /// Charge le registre complet d'un compte, avertissements expirés élagués.
    pub async fn shadowban_load_ledger(&self, user_id: &str) -> StrikeLedger {
        let now = Utc::now();
        let cutoff = (now - strike_ttl()).timestamp();
        let key = strikes_key(user_id);

        let raw: Vec<String> = {
            let mut c = self.conn.lock().await;
            // Élagage opportuniste : les expirés ne sont jamais relus ensuite.
            let _: Result<(), _> = c.zrembyscore(&key, f64::NEG_INFINITY, cutoff as f64).await;
            c.zrangebyscore(&key, cutoff as f64, f64::INFINITY)
                .await
                .unwrap_or_default()
        };

        let mut strikes = Vec::with_capacity(raw.len());
        for m in raw {
            match serde_json::from_str::<Strike>(&m) {
                Ok(s) => strikes.push(s),
                Err(e) => warn!(user_id, error = %e, "Avertissement illisible en Redis — ignoré"),
            }
        }

        let (manual_override, manual_expires_at) = self.shadowban_load_manual(user_id).await;

        StrikeLedger {
            user_id: user_id.to_string(),
            strikes,
            manual_override,
            manual_expires_at,
        }
    }

    /// Décision manuelle courante et sa date de fin (dérivée du TTL Redis).
    async fn shadowban_load_manual(
        &self,
        user_id: &str,
    ) -> (Option<ShadowbanLevel>, Option<DateTime<Utc>>) {
        let key = manual_key(user_id);
        let mut c = self.conn.lock().await;
        let raw: Option<String> = c.get(&key).await.ok().flatten();
        let Some(raw) = raw else { return (None, None) };
        let ttl: i64 = c.ttl(&key).await.unwrap_or(-1);
        drop(c);

        let level = ShadowbanLevel::from_label(&raw).unwrap_or_else(|| {
            error!(user_id, value = %raw, "Niveau manuel inconnu en Redis — traité comme Suppressed");
            ShadowbanLevel::Suppressed
        });
        // TTL négatif = pas d'expiration posée (clé permanente ou absente).
        let expires = (ttl > 0).then(|| Utc::now() + Duration::seconds(ttl));
        (Some(level), expires)
    }

    /// État de compte destiné au créateur concerné.
    pub async fn shadowban_account_status(&self, user_id: &str) -> super::strikes::AccountStatus {
        self.shadowban_load_ledger(user_id).await.status(Utc::now())
    }

    // ─── Lecture chaude ───────────────────────────────────────────────────────

    /// Niveaux effectifs pour un lot d'auteurs — chemin appelé sur chaque pool.
    ///
    /// Deux `MGET` au total, quel que soit le nombre d'auteurs. La version
    /// précédente faisait une lecture réseau **par auteur**, en série sous le
    /// verrou de la connexion : sur un pool de 200 candidats, 200 aller-retours
    /// bloquaient la constitution du feed.
    ///
    /// Le niveau retenu est le plus sévère entre la décision manuelle et le
    /// calcul par avertissements : un compte blanchi à la main reste blanchi
    /// seulement si l'on abaisse aussi ses avertissements (via
    /// `shadowban_clear_strikes`) — sinon la décision humaine ne pourrait que
    /// durcir. C'est volontaire côté chemin chaud, où l'on n'a pas le registre
    /// sous la main pour trancher ; `StrikeLedger::level` fait autorité partout
    /// ailleurs.
    pub async fn shadowban_load_levels(
        &self,
        user_ids: &[String],
    ) -> HashMap<String, ShadowbanLevel> {
        if user_ids.is_empty() {
            return HashMap::new();
        }

        let manual = self.mget_levels(user_ids, manual_key).await;
        let derived = self.mget_levels(user_ids, level_key).await;

        let mut out: HashMap<String, ShadowbanLevel> = HashMap::new();
        for (idx, uid) in user_ids.iter().enumerate() {
            let level = match (
                manual.get(idx).copied().flatten(),
                derived.get(idx).copied().flatten(),
            ) {
                (Some(m), Some(d)) => m.max(d),
                (Some(m), None) => m,
                (None, Some(d)) => d,
                (None, None) => continue,
            };
            if level != ShadowbanLevel::Clean {
                out.insert(uid.clone(), level);
            }
        }
        if !out.is_empty() {
            debug!(
                count = out.len(),
                pool = user_ids.len(),
                "Niveaux de restriction chargés"
            );
        }
        out
    }

    async fn mget_levels(
        &self,
        user_ids: &[String],
        key_of: fn(&str) -> String,
    ) -> Vec<Option<ShadowbanLevel>> {
        let mut cmd = redis::cmd("MGET");
        for uid in user_ids {
            cmd.arg(key_of(uid));
        }
        let raw: Vec<Option<String>> = {
            let mut c = self.conn.lock().await;
            cmd.query_async(&mut *c).await.unwrap_or_default()
        };

        user_ids
            .iter()
            .enumerate()
            .map(|(i, uid)| {
                let val = raw.get(i)?.as_deref()?;
                match ShadowbanLevel::from_label(val) {
                    Some(l) => Some(l),
                    None => {
                        // Donnée corrompue ou valeur d'une version future. On
                        // retient Suppressed plutôt que Clean : une lecture ratée
                        // ne doit pas lever une restriction en silence.
                        error!(user_id = %uid, value = val,
                               "Niveau de restriction inconnu en Redis — traité comme Suppressed");
                        Some(ShadowbanLevel::Suppressed)
                    }
                }
            })
            .collect()
    }
}
