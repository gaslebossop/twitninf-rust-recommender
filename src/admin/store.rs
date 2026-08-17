/// Admin store — opérations Redis pour la gestion des bans et shadowbans.
///
/// Clés Redis :
///   admin:shadowban:{user_id}        → niveau ("monitoring" | "suppressed" | "ghosted"),
///                                      avec TTL si la décision est temporaire
///   admin:shadowban:reason:{user_id} → raison (string optionnel)
///   admin:shadowban:users            → SET de tous les user_ids shadowbannis
///   admin:hardban:{user_id}          → "1"
///   admin:hardban:reason:{user_id}   → raison (string optionnel)
///   admin:hardban:users              → SET de tous les user_ids hard-banni
///   admin:algo:weights               → JSON des poids overridés manuellement
///
/// La lecture par lot des niveaux vit dans `shadowban::store` : elle combine
/// ces décisions manuelles avec le niveau dérivé du registre d'avertissements,
/// et c'est ce chemin-là qui alimente le scoring.
use std::collections::HashSet;

use redis::AsyncCommands;
use tracing::{debug, warn};

use crate::services::cache_manager::CacheManager;
use crate::shadowban::ShadowbanLevel;

use super::models::{AlgoWeights, BannedUser, ShadowbannedUser};

impl CacheManager {
    // ─── Shadowban ────────────────────────────────────────────────────────────

    /// Pose (ou lève) une décision manuelle de restriction.
    ///
    /// `expires_in_days` borne la décision dans le temps. Sans lui, la clé reste
    /// jusqu'à ce que quelqu'un pense à la retirer — ce qui, en pratique, veut
    /// dire jamais. Le chemin normal reste `shadowban_add_strike`, qui expire
    /// tout seul au bout de 90 jours ; cette fonction sert aux décisions prises
    /// en connaissance de cause.
    pub async fn admin_set_shadowban(
        &self,
        user_id: &str,
        level: ShadowbanLevel,
        reason: Option<&str>,
        expires_in_days: Option<u32>,
    ) {
        let mut c = self.conn.lock().await;

        if level == ShadowbanLevel::Clean {
            // Déshadowban : supprimer toutes les entrées
            let _: Result<(), _> = c.del(format!("admin:shadowban:{user_id}")).await;
            let _: Result<(), _> = c.del(format!("admin:shadowban:reason:{user_id}")).await;
            let _: Result<(), _> = c.srem("admin:shadowban:users", user_id).await;
            debug!(user_id, "Shadowban cleared (Clean)");
        } else {
            let level_str = level.label();
            let key = format!("admin:shadowban:{user_id}");
            let reason_key = format!("admin:shadowban:reason:{user_id}");
            match expires_in_days {
                Some(days) if days > 0 => {
                    let ttl = days as u64 * 86_400;
                    let _: Result<(), _> = c.set_ex(&key, level_str, ttl).await;
                    if let Some(r) = reason {
                        let _: Result<(), _> = c.set_ex(&reason_key, r, ttl).await;
                    }
                    debug!(user_id, level = level_str, days, "Shadowban set (temporaire)");
                }
                _ => {
                    let _: Result<(), _> = c.set(&key, level_str).await;
                    if let Some(r) = reason {
                        let _: Result<(), _> = c.set(&reason_key, r).await;
                    }
                    debug!(user_id, level = level_str, "Shadowban set (sans terme)");
                }
            }
            let _: Result<(), _> = c.sadd("admin:shadowban:users", user_id).await;
        }

        // Invalider le cache de profil pour forcer un rechargement
        let _: Result<(), _> = c.del(format!("twitninf:profile:{user_id}")).await;
    }

    /// Liste des comptes sous décision manuelle.
    ///
    /// Une décision temporaire expire côté clé de niveau mais laisse son
    /// identifiant dans le SET d'index : sans élagage, la liste d'administration
    /// afficherait indéfiniment des comptes qui ne sont plus restreints. On
    /// nettoie donc au passage.
    pub async fn admin_get_shadowbanned_users(&self) -> Vec<ShadowbannedUser> {
        let mut c = self.conn.lock().await;
        let user_ids: HashSet<String> = c.smembers("admin:shadowban:users").await.unwrap_or_default();
        drop(c);

        let mut result = Vec::new();
        let mut stale: Vec<String> = Vec::new();
        for uid in user_ids {
            let mut c2 = self.conn.lock().await;
            let level: Option<String> = c2.get(format!("admin:shadowban:{uid}")).await.ok().flatten();
            let reason: Option<String> = c2.get(format!("admin:shadowban:reason:{uid}")).await.ok().flatten();
            drop(c2);

            match level {
                Some(l) => result.push(ShadowbannedUser { user_id: uid, level: l, reason }),
                None => stale.push(uid),
            }
        }

        if !stale.is_empty() {
            let mut c3 = self.conn.lock().await;
            for uid in &stale {
                let _: Result<(), _> = c3.srem("admin:shadowban:users", uid).await;
            }
            drop(c3);
            debug!(count = stale.len(), "Décisions manuelles expirées retirées de l'index");
        }
        result
    }

    // ─── Hard ban ─────────────────────────────────────────────────────────────

    pub async fn admin_set_hard_ban(&self, user_id: &str, reason: Option<&str>) {
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.set(format!("admin:hardban:{user_id}"), "1").await;
        if let Some(r) = reason {
            let _: Result<(), _> = c.set(format!("admin:hardban:reason:{user_id}"), r).await;
        }
        let _: Result<(), _> = c.sadd("admin:hardban:users", user_id).await;
        // Invalider les recommandations du compte banni
        for mode in &["feed", "discover", "trending", "for_you"] {
            let _: Result<(), _> = c.del(format!("twitninf:reco:{user_id}:{mode}")).await;
        }
        debug!(user_id, "Hard ban set");
    }

    pub async fn admin_remove_hard_ban(&self, user_id: &str) {
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.del(format!("admin:hardban:{user_id}")).await;
        let _: Result<(), _> = c.del(format!("admin:hardban:reason:{user_id}")).await;
        let _: Result<(), _> = c.srem("admin:hardban:users", user_id).await;
        debug!(user_id, "Hard ban removed");
    }

    pub async fn admin_get_banned_users(&self) -> Vec<BannedUser> {
        let mut c = self.conn.lock().await;
        let user_ids: HashSet<String> = c.smembers("admin:hardban:users").await.unwrap_or_default();
        drop(c);

        let mut result = Vec::new();
        for uid in user_ids {
            let mut c2 = self.conn.lock().await;
            let reason: Option<String> = c2.get(format!("admin:hardban:reason:{uid}")).await.ok().flatten();
            drop(c2);
            result.push(BannedUser { user_id: uid, reason });
        }
        result
    }

    /// Retourne l'ensemble des user_ids hard-bannis (pour filtrage SQL)
    /// Résultat mis en cache 60 secondes dans la même clé Redis.
    pub async fn admin_get_hard_banned_set(&self) -> HashSet<String> {
        let mut c = self.conn.lock().await;
        match c.smembers::<_, HashSet<String>>("admin:hardban:users").await {
            Ok(set) => set,
            Err(e) => {
                warn!("Failed to load hard-ban set: {e}");
                HashSet::new()
            }
        }
    }

    /// Vérifie si un user est hard-banni
    pub async fn admin_is_hard_banned(&self, user_id: &str) -> bool {
        let mut c = self.conn.lock().await;
        let val: Option<String> = c.get(format!("admin:hardban:{user_id}")).await.ok().flatten();
        val.is_some()
    }

    // ─── Poids d'algo (override manuel) ──────────────────────────────────────

    pub async fn admin_save_weights(&self, weights: &AlgoWeights) {
        if let Ok(json) = serde_json::to_string(weights) {
            let mut c = self.conn.lock().await;
            let _: Result<(), _> = c.set("admin:algo:weights", json).await;
            debug!("Admin algo weights saved");
        }
    }

    pub async fn admin_load_weights(&self) -> Option<AlgoWeights> {
        let mut c = self.conn.lock().await;
        let json: Option<String> = c.get("admin:algo:weights").await.ok().flatten();
        json.and_then(|j| serde_json::from_str(&j).ok())
    }

    pub async fn admin_clear_weights(&self) {
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.del("admin:algo:weights").await;
        debug!("Admin algo weights cleared (back to auto/default)");
    }
}
