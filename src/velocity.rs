//! Throttle de vélocité — frein temporaire, automatique, sur des actions du
//! compte lui-même, jamais sur une décision de modération.
//!
//! Distinct du registre d'avertissements ([`crate::shadowban`]) : pas de
//! motif nommé, pas d'historique consultable, pas de recours — juste un
//! multiplicateur qui s'efface tout seul au bout d'une heure. Supprimer un
//! tweet, changer d'avatar ou de bio, publier en rafale, sont des gestes
//! LÉGITIMES la plupart du temps (corriger une coquille, mettre à jour son
//! profil) : les traiter comme un avertissement de modération — 90 jours,
//! motif affiché au créateur — serait disproportionné. Un frein d'une heure
//! absorbe le cas suspect (nettoyage après coup, bot qui change d'identité,
//! rafale de publication) sans laisser de trace durable sur un compte qui
//! n'a rien fait de mal.
//!
//! Déclencheurs (posés côté API, pas ici — ce module ne fait qu'appliquer) :
//! suppression d'un tweet, changement d'avatar, changement de bio, ou
//! publication de 10 tweets en moins de 10 minutes.

use std::collections::HashMap;

use redis::AsyncCommands;

use crate::services::cache_manager::CacheManager;

/// Durée du frein. Assez courte pour ne jamais ressembler à une sanction.
pub const VELOCITY_THROTTLE_TTL_SECS: u64 = 3600;

/// Multiplicateur appliqué au score final pendant le frein.
pub const VELOCITY_THROTTLE_MULTIPLIER: f64 = 0.5;

/// Fenêtre glissante de détection de rafale de publication.
pub const BURST_WINDOW_SECS: u64 = 600; // 10 minutes

/// Nombre de tweets dans la fenêtre au-delà duquel le frein se pose.
pub const BURST_THRESHOLD: i64 = 10;

fn key(user_id: &str) -> String {
    format!("velocity:throttle:{user_id}")
}

fn burst_key(user_id: &str) -> String {
    format!("velocity:posts:{user_id}")
}

impl CacheManager {
    /// Pose le frein sur un compte pour [`VELOCITY_THROTTLE_TTL_SECS`].
    ///
    /// Idempotent et sans cumul : une deuxième action déclenchante pendant que
    /// le frein est déjà actif le RECHARGE à une heure pleine plutôt que de
    /// l'empiler — un multiplicateur en dessous de 0,5 punirait un compte qui,
    /// par exemple, met à jour sa bio puis son avatar dans la foulée, alors que
    /// chacun des deux gestes pris seul est bénin.
    pub async fn set_velocity_throttle(&self, user_id: &str) {
        let mut c = self.conn.lock().await;
        let _: Result<(), _> = c.set_ex(key(user_id), "1", VELOCITY_THROTTLE_TTL_SECS).await;
    }

    /// Compte une publication dans la fenêtre de rafale, et pose le frein si
    /// le seuil ([`BURST_THRESHOLD`] tweets en [`BURST_WINDOW_SECS`]) est
    /// franchi. Retourne le compte courant, pour les logs de l'appelant.
    ///
    /// Contrairement aux trois autres déclencheurs (suppression, avatar, bio),
    /// UN post ne déclenche rien : c'est le rythme qui compte, pas le geste
    /// lui-même. `INCR` sur une clé à expiration coulissante — la première
    /// écriture pose le TTL, les suivantes l'allongent pas : la fenêtre reste
    /// bien de 10 minutes glissantes depuis le premier post compté, pas
    /// remise à zéro à chaque publication.
    pub async fn record_post_and_maybe_throttle(&self, user_id: &str) -> i64 {
        let key = burst_key(user_id);
        let count: i64 = {
            let mut c = self.conn.lock().await;
            let n: i64 = c.incr(&key, 1).await.unwrap_or(0);
            if n == 1 {
                let _: Result<(), _> = c.expire(&key, BURST_WINDOW_SECS as i64).await;
            }
            n
        };
        if count >= BURST_THRESHOLD {
            self.set_velocity_throttle(user_id).await;
        }
        count
    }

    /// Frein actif pour un lot de comptes — même patron que
    /// `shadowban_load_levels` : un seul `MGET` pour tout le pool de
    /// candidats, jamais un aller-retour Redis par auteur.
    pub async fn load_velocity_throttles(&self, user_ids: &[String]) -> HashMap<String, f64> {
        if user_ids.is_empty() {
            return HashMap::new();
        }
        let mut cmd = redis::cmd("MGET");
        for uid in user_ids {
            cmd.arg(key(uid));
        }
        let raw: Vec<Option<String>> = {
            let mut c = self.conn.lock().await;
            cmd.query_async(&mut *c).await.unwrap_or_default()
        };
        user_ids
            .iter()
            .enumerate()
            .filter_map(|(i, uid)| raw.get(i)?.as_ref().map(|_| (uid.clone(), VELOCITY_THROTTLE_MULTIPLIER)))
            .collect()
    }
}
