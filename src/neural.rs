//! Client du service `taste-model` — le modèle neuronal entraîné en continu.
//!
//! Le service tourne sur `127.0.0.1:3011` (unité systemd `taste-model`, dépôt
//! `taste-model/`). Il expose `POST /score` : on lui donne un lecteur et une
//! liste de tweets candidats, il rend cinq probabilités par tweet. Seule
//! `like` entre ici, en 7ᵉ terme du mélange.
//!
//! ── Trois garde-fous, dans cet ordre d'importance ───────────────────────────
//!
//! 1. **Interrupteur à chaud.** `admin:taste:enabled` dans Redis, lu à chaque
//!    requête en même temps que les poids d'algo (un `GET` de plus, sur un
//!    aller-retour qui existe déjà). Éteindre le modèle ne demande donc PAS de
//!    recompiler ni de redéployer le moteur — ce qui est la seule façon
//!    honnête d'assumer un branchement sur le fil de tout le monde.
//!
//! 2. **Délai maximal.** Au-delà de `NEURAL_TIMEOUT_MS`, on abandonne et on
//!    classe exactement comme avant. Un service lent ou mort ne peut pas
//!    ralentir le fil, seulement cesser d'y contribuer. Mesure faite sur ce
//!    VPS : 105 ms pour 50 candidats sur le lecteur le plus lourd de la base,
//!    43 à 78 ms pour un lecteur ordinaire.
//!
//! 3. **Échelle.** La valeur injectée est un `lift`, pas la probabilité brute.
//!    C'est le piège qui avait vidé les têtes multi-objectifs de leur effet en
//!    août : `blend_positive` fait une MOYENNE pondérée, donc une tête dont les
//!    valeurs vivent sur une plage étroite abaisse tous les scores d'autant et
//!    ne change aucun ordre, quels que soient les poids écrits. Même formule
//!    que `ml::objectives::Head::lift` — division par la moyenne courante, puis
//!    écrasement par `l / (l + 1)`, ce qui recentre la tête sur 0,5.
//!
//! ── Ce que le service ne fait PAS ───────────────────────────────────────────
//! Il n'écrit rien en base (sa connexion est ouverte en
//! `default_transaction_read_only=on`) et n'écoute que sur la boucle locale.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tracing::{debug, warn};

/// Au-delà, on renonce et on classe sans le modèle. Volontairement proche de la
/// latence mesurée du pire lecteur : mieux vaut perdre la contribution du
/// modèle que retarder le fil.
const NEURAL_TIMEOUT_MS: u64 = 250;

/// Plafond de candidats envoyés en une fois. Le vivier réel fait quelques
/// dizaines de tweets (78 candidats relevés en production le 2026-08-21), donc
/// ce plafond ne coupe rien aujourd'hui — il borne le pire cas.
const MAX_CANDIDATES: usize = 200;

/// Poids de la moyenne glissante servant au `lift`. Assez lent pour ne pas
/// suivre les à-coups d'une seule page, assez rapide pour suivre une dérive
/// réelle du modèle qui, lui, continue d'apprendre.
const MEAN_ALPHA: f64 = 0.01;

/// Sous ce nombre d'observations, le `lift` ne veut rien dire : la moyenne
/// courante est encore celle du prior. La tête reste alors muette, exactement
/// comme une tête `objectives` qui n'a pas atteint `MIN_SAMPLES`.
const MIN_OBSERVATIONS: u64 = 200;

#[derive(Deserialize)]
struct ScoreResponse {
    scores: HashMap<String, HeadScores>,
}

#[derive(Deserialize)]
struct HeadScores {
    like: f64,
}

#[derive(Default)]
pub struct NeuralStats {
    pub calls: u64,
    pub failures: u64,
    pub timeouts: u64,
    pub last_latency_ms: u64,
    pub last_error: Option<String>,
}

pub struct NeuralClient {
    http: reqwest::Client,
    url: String,
    key: String,
    /// Interrupteur à chaud, rafraîchi depuis Redis à chaque requête.
    enabled: AtomicBool,
    /// Le service est-il seulement configuré ? Sans URL, tout est inerte.
    configured: bool,
    mean: RwLock<f64>,
    observations: AtomicU64,
    stats: RwLock<NeuralStats>,
}

impl NeuralClient {
    pub fn from_env() -> Self {
        let url = std::env::var("TASTE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:3011".to_string());
        let key = std::env::var("TASTE_SERVICE_KEY").unwrap_or_default();
        let configured = !key.is_empty();
        if !configured {
            warn!(
                "TASTE_SERVICE_KEY absent : le modèle neuronal restera inerte \
                 (le classement se comporte comme avant son ajout)"
            );
        }
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_millis(NEURAL_TIMEOUT_MS))
                .build()
                .unwrap_or_default(),
            url,
            key,
            // Éteint par défaut. Un déploiement du moteur ne doit JAMAIS
            // allumer le modèle tout seul : c'est une décision qui se prend en
            // posant la clé Redis, pas en poussant du code.
            enabled: AtomicBool::new(false),
            configured,
            mean: RwLock::new(0.0),
            observations: AtomicU64::new(0),
            stats: RwLock::new(NeuralStats::default()),
        }
    }

    pub fn set_enabled(&self, on: bool) {
        let was = self.enabled.swap(on, Ordering::Relaxed);
        if was != on {
            debug!(enabled = on, "Modèle neuronal : bascule");
        }
    }

    pub fn is_active(&self) -> bool {
        self.configured && self.enabled.load(Ordering::Relaxed)
    }

    /// Assez d'observations pour que le `lift` soit interprétable ?
    pub fn is_warm(&self) -> bool {
        self.observations.load(Ordering::Relaxed) >= MIN_OBSERVATIONS
    }

    /// `p` ramenée sur l'échelle commune du mélange — voir l'en-tête.
    pub fn lift(&self, p: f64) -> f64 {
        let mean = self.mean.read().map(|m| *m).unwrap_or(0.0).max(1e-4);
        let l = p / mean;
        l / (l + 1.0)
    }

    fn observe(&self, p: f64) {
        if let Ok(mut m) = self.mean.write() {
            let n = self.observations.fetch_add(1, Ordering::Relaxed);
            // Moyenne d'amorçage exacte sur les premières valeurs, puis
            // exponentielle : démarrer une EMA à zéro donnerait un `lift`
            // gigantesque sur les toutes premières pages.
            *m = if n == 0 {
                p
            } else if n < MIN_OBSERVATIONS {
                (*m * n as f64 + p) / (n + 1) as f64
            } else {
                *m * (1.0 - MEAN_ALPHA) + p * MEAN_ALPHA
            };
        }
    }

    pub fn stats_snapshot(&self) -> (u64, u64, u64, u64, f64, bool, bool) {
        let s = self.stats.read();
        let (calls, failures, timeouts, last) = s
            .as_ref()
            .map(|s| (s.calls, s.failures, s.timeouts, s.last_latency_ms))
            .unwrap_or((0, 0, 0, 0));
        (
            calls,
            failures,
            timeouts,
            last,
            self.mean.read().map(|m| *m).unwrap_or(0.0),
            self.is_active(),
            self.is_warm(),
        )
    }

    /// Le `/stats` du service, relayé tel quel pour le panneau admin.
    ///
    /// Relayé par le MOTEUR et pas lu directement par le navigateur : le
    /// service n'écoute que sur `127.0.0.1`, il est donc injoignable depuis un
    /// poste d'admin. Le moteur est déjà exposé et déjà authentifié — c'est le
    /// seul chemin qui n'ouvre rien de nouveau.
    ///
    /// Ignore l'interrupteur : on veut pouvoir regarder ce que fait le service
    /// même quand il est débranché du classement. C'est précisément le moment
    /// où on a besoin de le voir.
    pub async fn service_stats(&self) -> Option<serde_json::Value> {
        if !self.configured {
            return None;
        }
        let res = self
            .http
            .get(format!("{}/stats", self.url))
            .header("X-Service-Key", &self.key)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .ok()?;
        if !res.status().is_success() {
            return None;
        }
        res.json().await.ok()
    }

    /// Probabilité `like` par tweet. Vide si éteint, non configuré, ou en
    /// échec — l'appelant ne distingue pas les trois cas, et n'a pas à le
    /// faire : dans tous, le classement retombe sur son comportement d'avant.
    pub async fn scores(&self, user_id: &str, tweet_ids: &[String]) -> HashMap<String, f64> {
        if !self.is_active() || tweet_ids.is_empty() {
            return HashMap::new();
        }
        let ids: Vec<&String> = tweet_ids.iter().take(MAX_CANDIDATES).collect();
        let body = serde_json::json!({ "user_id": user_id, "tweet_ids": ids });

        let t0 = Instant::now();
        let res = self
            .http
            .post(format!("{}/score", self.url))
            .header("X-Service-Key", &self.key)
            .json(&body)
            .send()
            .await;
        let elapsed = t0.elapsed().as_millis() as u64;

        let mut record_failure = |timeout: bool, err: String| {
            if let Ok(mut s) = self.stats.write() {
                s.calls += 1;
                s.failures += 1;
                if timeout {
                    s.timeouts += 1;
                }
                s.last_latency_ms = elapsed;
                s.last_error = Some(err);
            }
        };

        let resp = match res {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let st = r.status();
                record_failure(false, format!("HTTP {}", st));
                warn!(status = %st, "Modèle neuronal : réponse non-200, classement inchangé");
                return HashMap::new();
            }
            Err(e) => {
                let timeout = e.is_timeout();
                record_failure(timeout, e.to_string());
                warn!(error = %e, timeout, elapsed_ms = elapsed,
                      "Modèle neuronal injoignable, classement inchangé");
                return HashMap::new();
            }
        };

        let parsed: ScoreResponse = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                record_failure(false, e.to_string());
                warn!(error = %e, "Modèle neuronal : réponse illisible");
                return HashMap::new();
            }
        };

        if let Ok(mut s) = self.stats.write() {
            s.calls += 1;
            s.last_latency_ms = elapsed;
        }
        let out: HashMap<String, f64> = parsed
            .scores
            .into_iter()
            .map(|(id, h)| (id, h.like.clamp(0.0, 1.0)))
            .collect();
        for p in out.values() {
            self.observe(*p);
        }
        debug!(n = out.len(), elapsed_ms = elapsed, "Modèle neuronal : scores reçus");
        out
    }
}

impl Default for NeuralClient {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> NeuralClient {
        NeuralClient {
            http: reqwest::Client::new(),
            url: "http://127.0.0.1:1".into(),
            key: "k".into(),
            enabled: AtomicBool::new(true),
            configured: true,
            mean: RwLock::new(0.0),
            observations: AtomicU64::new(0),
            stats: RwLock::new(NeuralStats::default()),
        }
    }

    #[test]
    fn lift_recentre_la_tete_sur_un_demi() {
        // Une tête dont toutes les valeurs valent la moyenne doit rendre 0,5,
        // quelle que soit la rareté de l'événement. C'est toute la raison
        // d'être du `lift` : sans lui, une tête qui prédit 0,03 ne peut pas
        // peser dans une moyenne où les règles valent 0,5.
        for p in [0.002_f64, 0.03, 0.3, 0.8] {
            let c = client();
            for _ in 0..300 {
                c.observe(p);
            }
            assert!((c.lift(p) - 0.5).abs() < 1e-6, "p={p} lift={}", c.lift(p));
        }
    }

    #[test]
    fn lift_monotone_et_borne() {
        let c = client();
        for _ in 0..300 {
            c.observe(0.2);
        }
        assert!(c.lift(0.4) > c.lift(0.2));
        assert!(c.lift(0.1) < c.lift(0.2));
        assert!((0.0..=1.0).contains(&c.lift(0.99)));
        assert!((0.0..=1.0).contains(&c.lift(0.0)));
    }

    #[test]
    fn moyenne_exacte_avant_le_seuil_puis_exponentielle() {
        let c = client();
        c.observe(0.4);
        c.observe(0.6);
        // Moyenne arithmétique tant qu'on est sous le seuil : démarrer une EMA
        // à zéro donnerait un lift gigantesque sur les premières pages.
        assert!((*c.mean.read().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tete_muette_tant_qu_elle_n_a_pas_assez_vu() {
        let c = client();
        assert!(!c.is_warm());
        for _ in 0..MIN_OBSERVATIONS {
            c.observe(0.3);
        }
        assert!(c.is_warm());
    }

    #[tokio::test]
    async fn service_injoignable_rend_une_carte_vide() {
        // Le contrat qui compte : une panne du service ne doit jamais faire
        // autre chose que retirer sa contribution.
        let c = client();
        let ids = vec!["a".to_string(), "b".to_string()];
        let out = c.scores("u", &ids).await;
        assert!(out.is_empty());
        assert_eq!(c.stats.read().unwrap().failures, 1);
    }

    #[tokio::test]
    async fn eteint_n_appelle_pas_le_service() {
        let c = client();
        c.set_enabled(false);
        let out = c.scores("u", &["a".to_string()]).await;
        assert!(out.is_empty());
        // Aucun appel réseau tenté : le compteur reste à zéro.
        assert_eq!(c.stats.read().unwrap().calls, 0);
    }
}
