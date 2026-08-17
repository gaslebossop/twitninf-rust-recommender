//! Boost temps réel, à l'intérieur d'une même session de scroll.
//!
//! ── Pourquoi ce fichier existait sans jamais tourner ────────────────────────
//! Cette version d'origine était écrite contre `deadpool_redis::Pool` — un
//! type de connexion que le reste du moteur n'utilise nulle part.
//! `CacheManager` (ce que `AppState` transporte réellement) tient une
//! `MultiplexedConnection` brute. Le fichier n'était même pas déclaré dans
//! `services/mod.rs` : incompatible avec le reste ET absent du binaire, donc
//! invisible à la fois au compilateur et à l'exécution.
//!
//! ── Ce qui a aussi été simplifié en le rebranchant ──────────────────────────
//! La version d'origine boostait aussi par HASHTAG, mais aucune chaîne de
//! hashtag n'atteint `RawTweet` (seul `hashtag_count`, un entier, est chargé) ni
//! `TrackInteractionRequest` (aucun champ hashtags) — ce volet n'a jamais pu
//! fonctionner, avec ou sans le bon type de connexion. Retiré plutôt que
//! rebranché à moitié ; le boost par AUTEUR, lui, est immédiatement actionnable
//! avec ce que la requête de tracking porte déjà (`author_id`).
//!
//! ── Ce que ça change dans le classement ─────────────────────────────────────
//! Sans ce fichier, entre deux tweets d'un même auteur qu'on vient d'aimer, le
//! moteur ne réagit qu'au prochain rechargement de profil ou à l'expiration du
//! cache de feed (30 à 180 s) — jamais DANS la page qu'on est en train de lire.
//! Avec lui, l'auteur qu'on vient d'aimer est mis en avant pour les 30 minutes
//! qui suivent, dès la page suivante de la même session.

use redis::AsyncCommands;
use tracing::{debug, warn};

use super::cache_manager::CacheManager;

const CLICK_BOOST: f64 = 0.15;
const SKIP_PENALTY: f64 = -0.08;
const BOOST_TTL_SECS: u64 = 1800; // 30 minutes

fn is_valid_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

fn author_key(user_id: &str, author_id: &str) -> String {
    format!("feedback:author:{user_id}:{author_id}")
}

impl CacheManager {
    /// Enregistre une réaction (clic/engagement ou skip) envers un auteur
    /// précis, pour CE lecteur. Appelé depuis `track_handler` sur toute
    /// interaction positive ou sur un skip explicite — jamais sur une simple
    /// vue, qui ne dit rien de l'intention.
    pub async fn record_author_feedback(&self, user_id: &str, author_id: &str, clicked: bool) {
        if !is_valid_uuid(user_id) || !is_valid_uuid(author_id) {
            warn!(user_id, author_id, "feedback: UUID invalide — ignoré");
            return;
        }
        let delta = if clicked { CLICK_BOOST } else { SKIP_PENALTY };
        let key = author_key(user_id, author_id);
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.set_ex(&key, delta.to_string(), BOOST_TTL_SECS).await;
        debug!(
            user_id,
            author_id, clicked, delta, "Feedback temps réel enregistré"
        );
    }

    /// Boosts actifs pour un lot d'auteurs candidats — même patron que
    /// `shadowban_load_levels`/`load_velocity_throttles` : un seul `MGET`
    /// pour tout le pool, jamais un aller-retour par auteur.
    pub async fn load_realtime_author_boosts(
        &self,
        user_id: &str,
        author_ids: &[String],
    ) -> std::collections::HashMap<String, f64> {
        if author_ids.is_empty() {
            return std::collections::HashMap::new();
        }
        let mut cmd = redis::cmd("MGET");
        for aid in author_ids {
            cmd.arg(author_key(user_id, aid));
        }
        let raw: Vec<Option<String>> = {
            let mut c = self.conn.lock().await;
            cmd.query_async(&mut *c).await.unwrap_or_default()
        };
        author_ids
            .iter()
            .enumerate()
            .filter_map(|(i, aid)| {
                let val = raw.get(i)?.as_deref()?;
                val.parse::<f64>().ok().map(|d| (aid.clone(), d))
            })
            .collect()
    }
}
