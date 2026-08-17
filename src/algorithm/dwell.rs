//! Normalisation du temps de lecture (« dwell time »).
//!
//! ── Pourquoi le temps BRUT ne veut rien dire ────────────────────────────────
//! Le handler de suivi convertissait le temps passé en bonus par paliers fixes
//! sur la valeur brute : `> 10 s → 0.5`, `> 3 s → 0.2`, sinon `0`. C'est le
//! défaut que toute la littérature du domaine décrit comme le piège numéro un :
//! le temps passé sur un contenu est confondu avec sa LONGUEUR, pas seulement
//! avec l'intérêt qu'on lui porte.
//!
//! Concrètement, avec des paliers bruts :
//!   * un pavé de 280 caractères qu'on lit sans plaisir dépasse 10 s et récolte
//!     le bonus maximal ;
//!   * un tweet de 40 caractères qu'on trouve excellent ne peut PAS matérielle-
//!     ment durer 10 s — il plafonne à 0.2, comme un contenu survolé ;
//!   * une vidéo de 8 s regardée trois fois en boucle vaut autant qu'une vidéo
//!     de 3 min abandonnée au quart.
//! Le classement apprend donc « le public aime les contenus longs », ce qui est
//! une propriété du chronomètre, pas du public. C'est ce biais de durée que les
//! travaux du domaine corrigent (D2Q le fait par quantiles à l'intérieur de
//! tranches de durée ; le « watch ratio » plus simplement en divisant par la
//! durée ; voir les références en fin de fichier).
//!
//! ── Ce qui est fait ici ─────────────────────────────────────────────────────
//! On applique la version simple et robuste : rapporter le temps observé au
//! temps ATTENDU pour ce contenu-là, puis borner. Le quantile par tranche de
//! durée (D2Q) serait plus fin, mais demande une distribution historique par
//! contenu qu'on ne stocke nulle part — à reconsidérer le jour où ces
//! distributions existent.
//!
//! Deux conséquences voulues :
//!   * un contenu CONSOMMÉ EN ENTIER vaut pareil qu'il soit court ou long ;
//!   * un temps très inférieur à l'attendu devient un signal NÉGATIF, et pas
//!     seulement une absence de signal — s'arrêter une demi-seconde sur un
//!     contenu, c'est le refuser.

/// Nature du contenu, qui décide du temps de consommation attendu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DwellMedia {
    Text,
    Image,
    Video,
}

/// Ce qu'il faut savoir du contenu pour interpréter un temps passé.
#[derive(Debug, Clone)]
pub struct DwellContext {
    pub media: DwellMedia,
    /// Longueur du texte affiché, en caractères.
    pub content_chars: u32,
    /// Durée réelle de la vidéo, quand le lecteur a pu la remonter.
    pub video_duration_ms: Option<u32>,
}

/// Temps d'orientation incompressible : reconnaître l'auteur, cadrer l'image.
/// En dessous, on n'a rien consommé du tout, quel que soit le contenu.
const ORIENT_MS: f64 = 700.0;

/// Vitesse de lecture retenue, en caractères par seconde.
///
/// La lecture soutenue d'un adulte tourne autour de 200 mots/minute, soit
/// ~20 car/s en français (mot moyen + espace ≈ 6 caractères). Dans un fil on
/// SURVOLE plutôt qu'on ne lit : on retient un peu plus rapide, pour ne pas
/// exiger d'un texte long un temps que personne ne lui accorde jamais.
const READING_CHARS_PER_SEC: f64 = 25.0;

/// Temps attendu devant une image seule.
const IMAGE_EXPECTED_MS: f64 = 2_500.0;

/// Repli quand la durée d'une vidéo n'est pas connue (client ancien, ou lecture
/// coupée avant que la durée ne remonte).
const VIDEO_FALLBACK_MS: f64 = 8_000.0;

/// Une vidéo longue n'a pas à être regardée en entier pour valoir un signal
/// positif : au-delà, on considère l'intérêt démontré.
const VIDEO_EXPECTED_CAP_MS: f64 = 30_000.0;

/// Plancher de temps attendu — évite qu'un contenu minuscule rende le rapport
/// explosif (division par presque zéro).
const EXPECTED_FLOOR_MS: f64 = 1_200.0;

/// En dessous de cette part du temps attendu, on considère le contenu REFUSÉ.
const SKIP_RATIO: f64 = 0.20;

/// Poids le plus négatif qu'un survol puisse produire.
///
/// Volontairement du même ordre que `InteractionType::Skip` (-0.5) : c'est le
/// même geste, constaté au chronomètre plutôt que déclaré.
const SKIP_PENALTY: f64 = -0.45;

/// Bonus maximal d'un contenu longuement consommé.
///
/// Calibré pour qu'un contenu consommé en entier pèse à peu près comme un like
/// (`InteractionType::Like` = 1.0) une fois ajouté au poids d'une vue (0.2), et
/// jamais plus : un temps de lecture reste une préférence DEVINÉE, il ne doit
/// pas dominer un geste explicite.
const MAX_BONUS: f64 = 0.85;

/// Rapport au-delà duquel le bonus ne bouge plus (relecture, boucle vidéo).
const RATIO_CAP: f64 = 3.0;

/// Temps attendu pour consommer ce contenu, en millisecondes.
pub fn expected_consumption_ms(ctx: &DwellContext) -> f64 {
    let raw = match ctx.media {
        // La durée de la vidéo EST le temps attendu : le rapport devient un
        // taux de complétion, ce qui met une vidéo de 5 s et une de 2 min sur
        // la même échelle.
        DwellMedia::Video => ctx
            .video_duration_ms
            .map(|d| d as f64)
            .unwrap_or(VIDEO_FALLBACK_MS)
            .min(VIDEO_EXPECTED_CAP_MS),
        DwellMedia::Image => {
            IMAGE_EXPECTED_MS + ctx.content_chars as f64 / READING_CHARS_PER_SEC * 1000.0
        }
        DwellMedia::Text => ORIENT_MS + ctx.content_chars as f64 / READING_CHARS_PER_SEC * 1000.0,
    };
    raw.max(EXPECTED_FLOOR_MS)
}

/// Poids d'un temps de lecture, rapporté à ce que ce contenu-là demandait.
///
/// Sans contexte (client qui n'envoie pas encore la nature du contenu), on
/// retombe sur l'ancien comportement par paliers bruts : dégradé, mais jamais
/// pire qu'avant.
pub fn dwell_weight(dwell_ms: u32, ctx: Option<&DwellContext>) -> f64 {
    let Some(ctx) = ctx else {
        return legacy_dwell_bonus(dwell_ms);
    };

    let expected = expected_consumption_ms(ctx);
    let ratio = (dwell_ms as f64 / expected).min(RATIO_CAP);

    if ratio < SKIP_RATIO {
        // Décroît linéairement jusqu'au refus franc : s'arrêter juste sous le
        // seuil n'est pas la même chose que ne pas s'arrêter du tout.
        let severity = 1.0 - (ratio / SKIP_RATIO);
        return SKIP_PENALTY * severity;
    }

    // Saturation douce : au-delà de la consommation complète, chaque seconde
    // supplémentaire compte de moins en moins. Pas de palier, donc pas de
    // marche arbitraire où une milliseconde change le poids du simple au double.
    let normalized = (ratio - SKIP_RATIO) / (RATIO_CAP - SKIP_RATIO);
    MAX_BONUS * (normalized / (normalized + 0.45))
}

/// Ancien comportement, conservé pour les clients qui n'envoient pas de
/// contexte. Ne pas l'étendre : il porte exactement le biais de durée décrit
/// en tête de fichier.
fn legacy_dwell_bonus(dwell_ms: u32) -> f64 {
    if dwell_ms > 10_000 {
        0.5
    } else if dwell_ms > 3_000 {
        0.2
    } else {
        0.0
    }
}

// Références :
//   * Yi et al., « Beyond Clicks: Dwell Time for Personalization », RecSys 2014
//     (meilleur article) — normaliser le temps pour le rendre comparable d'un
//     contenu et d'un contexte à l'autre.
//   * Zhan et al., « Deconfounding Duration Bias in Watch-time Prediction for
//     Video Recommendation », KDD 2022 (D2Q) — quantiles par tranche de durée.
//   * « Counteracting Duration Bias in Video Recommendation via Counterfactual
//     Watch Time » (2024) — pourquoi un contenu lu EN ENTIER sature le signal.

#[cfg(test)]
mod tests {
    use super::*;

    fn text(chars: u32) -> DwellContext {
        DwellContext {
            media: DwellMedia::Text,
            content_chars: chars,
            video_duration_ms: None,
        }
    }
    fn video(duration_ms: u32) -> DwellContext {
        DwellContext {
            media: DwellMedia::Video,
            content_chars: 0,
            video_duration_ms: Some(duration_ms),
        }
    }

    #[test]
    fn un_contenu_consomme_en_entier_vaut_pareil_court_ou_long() {
        // C'est TOUT l'objet de la normalisation : le tweet court lu en entier
        // ne doit plus être pénalisé face au pavé lu en entier.
        let court = text(40);
        let long = text(280);
        let w_court = dwell_weight(expected_consumption_ms(&court) as u32, Some(&court));
        let w_long = dwell_weight(expected_consumption_ms(&long) as u32, Some(&long));
        assert!(
            (w_court - w_long).abs() < 1e-9,
            "court={w_court}, long={w_long}"
        );
    }

    #[test]
    fn l_ancien_calcul_privilegiait_le_texte_long_a_interet_egal() {
        // Vérifie que le défaut corrigé existait bien : à consommation complète
        // des deux côtés, les paliers bruts donnaient des poids différents.
        let court = expected_consumption_ms(&text(40)) as u32;
        let long = expected_consumption_ms(&text(280)) as u32;
        assert!(
            legacy_dwell_bonus(long) > legacy_dwell_bonus(court),
            "le biais de durée devait exister dans l'ancien calcul"
        );
    }

    #[test]
    fn un_survol_est_un_signal_negatif() {
        let ctx = text(200);
        let expected = expected_consumption_ms(&ctx);
        let w = dwell_weight((expected * 0.02) as u32, Some(&ctx));
        assert!(w < 0.0, "un survol doit peser négativement, obtenu {w}");
        assert!(
            w >= SKIP_PENALTY,
            "jamais en dessous du plancher, obtenu {w}"
        );
    }

    #[test]
    fn le_poids_croit_avec_le_temps_puis_sature() {
        let ctx = text(140);
        let expected = expected_consumption_ms(&ctx);
        let quart = dwell_weight((expected * 0.25) as u32, Some(&ctx));
        let entier = dwell_weight(expected as u32, Some(&ctx));
        let double = dwell_weight((expected * 2.0) as u32, Some(&ctx));
        let dix_fois = dwell_weight((expected * 10.0) as u32, Some(&ctx));

        assert!(quart < entier, "quart={quart} entier={entier}");
        assert!(entier < double, "entier={entier} double={double}");
        assert!(double <= MAX_BONUS);
        assert!(
            (dix_fois - double).abs() < 0.2,
            "au-delà du plafond le poids ne doit plus bouger : double={double} dix_fois={dix_fois}"
        );
    }

    #[test]
    fn une_courte_video_bouclee_vaut_une_longue_video_regardee() {
        // Une vidéo de 6 s vue deux fois est au moins aussi appréciée qu'une
        // vidéo de 60 s regardée en entier — l'ancien calcul disait l'inverse.
        let courte = video(6_000);
        let longue = video(60_000);
        let w_courte = dwell_weight(12_000, Some(&courte));
        let w_longue = dwell_weight(60_000, Some(&longue));
        assert!(w_courte >= w_longue, "courte={w_courte}, longue={w_longue}");
    }

    #[test]
    fn le_poids_reste_borne_dans_tous_les_cas() {
        let cases = [text(0), text(1), text(5_000), video(1), video(600_000)];
        for ctx in &cases {
            for ms in [0u32, 1, 500, 5_000, 60_000, 3_600_000] {
                let w = dwell_weight(ms, Some(ctx));
                assert!(
                    w >= SKIP_PENALTY && w <= MAX_BONUS,
                    "poids hors bornes: {w} ({ctx:?}, {ms}ms)"
                );
                assert!(w.is_finite(), "poids non fini pour {ctx:?} / {ms}ms");
            }
        }
    }

    #[test]
    fn sans_contexte_on_retombe_sur_l_ancien_comportement() {
        assert_eq!(dwell_weight(12_000, None), 0.5);
        assert_eq!(dwell_weight(5_000, None), 0.2);
        assert_eq!(dwell_weight(1_000, None), 0.0);
    }
}
