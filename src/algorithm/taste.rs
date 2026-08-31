//! D10 — affinité sémantique entre le goût du lecteur et le contenu.
//!
//! ── Pourquoi cette dimension existe ─────────────────────────────────────────
//! `UserProfile::taste_vector` — la moyenne des plongements des tweets que ce
//! lecteur a aimés ou longuement lus — est le signal le plus personnel dont le
//! moteur dispose. Il ne demande aucun entraînement, il parle dès le cinquième
//! like, et il dit quelque chose qu'aucune autre dimension ne dit : « ce texte
//! ressemble-t-il à ce que CETTE personne consomme ? »
//!
//! Il était calculé à chaque profil, stocké, et **ne servait qu'à deux choses**
//! — cibler les publicités, et renforcer l'onglet Explorer. Le fil principal
//! (`for_you`), celui que les applications demandent réellement, ne le
//! consultait jamais. La dimension la plus ciblée du moteur était éteinte sur
//! la seule surface qui compte.
//!
//! ── Pourquoi un RANG, et pas la similarité brute ────────────────────────────
//! Relevé en production : la similarité cosinus médiane entre un lecteur et un
//! candidat quelconque vaut ~0,65. Ce n'est pas un hasard — deux textes courts
//! en français partagent énormément de structure, et le plongement le voit.
//!
//! Verser 0,65 dans le score de presque tous les candidats revient à ajouter
//! une CONSTANTE : le classement ne bouge pas d'un rang. Ce qui discrimine,
//! c'est l'écart d'un candidat au reste du vivier, pas sa valeur absolue.
//!
//! On classe donc les candidats entre eux et on rend la position, de 0 (le plus
//! éloigné du goût du lecteur) à 1 (le plus proche). Aucun seuil absolu à
//! régler, et la dimension garde le même pouvoir de séparation que le vivier
//! soit globalement proche ou globalement lointain du lecteur.

use std::collections::HashMap;

/// Valeur rendue quand on ne sait pas : lecteur sans vecteur de goût, tweet
/// sans plongement, vivier trop petit pour classer.
///
/// 0,5 et non 0 : une dimension qui ne sait rien ne doit pas PÉNALISER. Un
/// tweet publié il y a trois minutes n'a pas encore son plongement — le
/// rétrograder pour ça reviendrait à filtrer la fraîcheur, exactement ce que
/// D4 s'efforce de favoriser.
pub const NEUTRAL: f64 = 0.5;

/// Nombre minimal de mesures pour qu'un classement veuille dire quelque chose.
///
/// À deux candidats, la « position » ne peut valoir que 0 ou 1 : la dimension
/// deviendrait un interrupteur qui écarte l'un des deux de toute la largeur de
/// son poids, sur un écart de similarité peut-être infime.
pub const MIN_POOL: usize = 5;

/// Position de chaque tweet dans le vivier, de 0 (le plus loin du goût du
/// lecteur) à 1 (le plus proche).
///
/// Les ex æquo reçoivent la même position — celle du milieu de leur groupe —
/// pour que des similarités identiques ne se départagent pas au hasard de
/// l'ordre de la table de hachage.
pub fn affinity_ranks(similarities: &HashMap<String, f64>) -> HashMap<String, f64> {
    if similarities.len() < MIN_POOL {
        return HashMap::new();
    }

    // Tri par similarité croissante. `total_cmp` plutôt que `partial_cmp` :
    // un NaN échappé de la base rendrait l'ordre incohérent et le tri lui-même
    // indéfini, au lieu de simplement se ranger à une extrémité.
    let mut ordered: Vec<(&String, f64)> = similarities.iter().map(|(id, s)| (id, *s)).collect();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1));

    let last = ordered.len() - 1;
    let mut ranks = HashMap::with_capacity(ordered.len());
    let mut i = 0usize;
    while i < ordered.len() {
        // Étendue du groupe d'ex æquo qui commence en `i`.
        let mut j = i;
        while j + 1 < ordered.len() && ordered[j + 1].1.total_cmp(&ordered[i].1).is_eq() {
            j += 1;
        }
        // Position du MILIEU du groupe, partagée par tous ses membres.
        let position = ((i + j) as f64 / 2.0) / last as f64;
        for (id, _) in &ordered[i..=j] {
            ranks.insert((*id).clone(), position.clamp(0.0, 1.0));
        }
        i = j + 1;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sims(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(id, s)| ((*id).to_string(), *s)).collect()
    }

    #[test]
    fn le_plus_proche_prend_un_et_le_plus_loin_zero() {
        let r = affinity_ranks(&sims(&[
            ("a", 0.20),
            ("b", 0.40),
            ("c", 0.60),
            ("d", 0.80),
            ("e", 0.90),
        ]));
        assert_eq!(r["a"], 0.0);
        assert_eq!(r["e"], 1.0);
        assert!(r["c"] > r["b"] && r["d"] > r["c"]);
    }

    /// Le point de toute la dimension : un vivier globalement proche du lecteur
    /// et un vivier globalement lointain doivent produire le MÊME étalement.
    /// C'est ce que la similarité brute ne fait pas — elle rendrait 0,64…0,68
    /// dans un cas et 0,10…0,90 dans l'autre.
    #[test]
    fn l_etalement_ne_depend_pas_du_niveau_absolu() {
        let serre = affinity_ranks(&sims(&[
            ("a", 0.640),
            ("b", 0.650),
            ("c", 0.660),
            ("d", 0.670),
            ("e", 0.680),
        ]));
        let large = affinity_ranks(&sims(&[
            ("a", 0.10),
            ("b", 0.30),
            ("c", 0.50),
            ("d", 0.70),
            ("e", 0.90),
        ]));
        for id in ["a", "b", "c", "d", "e"] {
            assert_eq!(serre[id], large[id], "tweet {id}");
        }
    }

    #[test]
    fn les_ex_aequo_partagent_la_meme_position() {
        let r = affinity_ranks(&sims(&[
            ("a", 0.10),
            ("b", 0.50),
            ("c", 0.50),
            ("d", 0.50),
            ("e", 0.90),
        ]));
        assert_eq!(r["b"], r["c"]);
        assert_eq!(r["c"], r["d"]);
        // Milieu du groupe {1,2,3} sur une échelle de 0 à 4.
        assert!((r["b"] - 0.5).abs() < 1e-12);
    }

    /// Toutes les similarités identiques : personne ne se distingue, tout le
    /// monde doit atterrir au même endroit — pas un classement arbitraire tiré
    /// de l'ordre de parcours de la table.
    #[test]
    fn un_vivier_plat_ne_departage_personne() {
        let r = affinity_ranks(&sims(&[
            ("a", 0.65),
            ("b", 0.65),
            ("c", 0.65),
            ("d", 0.65),
            ("e", 0.65),
        ]));
        let premiere = r["a"];
        for id in ["b", "c", "d", "e"] {
            assert_eq!(r[id], premiere);
        }
    }

    #[test]
    fn un_vivier_trop_petit_ne_rend_rien() {
        for n in 0..MIN_POOL {
            let pairs: Vec<(String, f64)> = (0..n)
                .map(|i| (format!("t{i}"), i as f64 / 10.0))
                .collect();
            let map: HashMap<String, f64> = pairs.into_iter().collect();
            assert!(
                affinity_ranks(&map).is_empty(),
                "{n} candidats devraient rester sans classement"
            );
        }
    }

    #[test]
    fn les_positions_restent_dans_zero_un() {
        let mut pairs = Vec::new();
        for i in 0..200 {
            // Valeurs volontairement hors de [0,1] et non triées.
            pairs.push((format!("t{i}"), ((i * 37) % 200) as f64 - 50.0));
        }
        let map: HashMap<String, f64> = pairs.into_iter().collect();
        for (id, position) in affinity_ranks(&map) {
            assert!(
                (0.0..=1.0).contains(&position),
                "{id} hors bornes : {position}"
            );
        }
    }
}
