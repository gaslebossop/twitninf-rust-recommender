//! Analyse du texte d'un tweet, faite UNE fois à l'entrée du pipeline.
//!
//! ── Pourquoi ce module existe ───────────────────────────────────────────────
//! Ce calcul vivait en ligne dans `map_rows`, où il n'était ni testable ni
//! mesurable : il fallait une `tokio_postgres::Row` pour l'appeler. Il tourne
//! pourtant sur CHAQUE candidat — 1700 par recommandation — et il est le seul
//! poste du pipeline qui parcourt le texte caractère par caractère.
//!
//! ── Ce qui a changé en le sortant ───────────────────────────────────────────
//! Le texte était balayé **six fois** : une conversion en minuscules, un
//! passage pour les émojis, un pour les `!`, un pour les `?`, deux recherches
//! de sous-chaîne pour les URLs, puis un découpage en mots qui allouait
//! jusqu'à cinquante `String` par tweet — 85 000 allocations par
//! recommandation, pour des mots dont un seul consommateur se sert (le
//! détecteur de contenu poubelle, qui les compte et les déduplique).
//!
//! Ici : un seul passage sur les caractères, et les mots gardés comme bornes
//! dans la chaîne déjà en minuscules au lieu d'être recopiés.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Longueur minimale d'un mot retenu. En dessous, ce sont des articles et des
/// prépositions : ni le comptage ni la déduplication n'en tirent quoi que ce
/// soit.
const MIN_WORD_LEN: usize = 4;

/// Nombre maximum de mots retenus par tweet.
const MAX_WORDS: usize = 50;

/// Traits textuels d'un tweet, extraits en un passage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentFeatures {
    /// Le texte en minuscules. Conservé (et non recalculé au scoring) parce
    /// que D2 et D8 y cherchent les centres d'intérêt du lecteur à chaque
    /// recommandation servie, et que `words` y pointe.
    pub lower: String,
    /// Bornes des mots retenus DANS `lower`.
    ///
    /// Des bornes et non des `String` : les mots ne servent qu'à être comptés
    /// et dédupliqués, jamais conservés au-delà du tweet. Cinquante
    /// allocations par candidat pour ça, c'était le poste d'allocation le plus
    /// lourd de tout le pipeline.
    pub word_spans: Vec<Range<u32>>,
    pub emoji_count: i32,
    pub exclamation_count: i32,
    pub question_count: i32,
    pub url_count: i32,
}

impl ContentFeatures {
    /// Les mots retenus, empruntés à `lower`.
    #[inline]
    pub fn words(&self) -> impl Iterator<Item = &str> + '_ {
        self.word_spans
            .iter()
            .map(|span| &self.lower[span.start as usize..span.end as usize])
    }

    #[inline]
    pub fn word_count(&self) -> usize {
        self.word_spans.len()
    }
}

/// Un caractère fait-il partie des blocs émoji ?
///
/// Le seuil d'origine (`> 0x1F000`) manquait ❤ ✅ ⭐ et comptait à tort des
/// idéogrammes CJK situés plus haut. Ces bornes-ci sont celles des blocs
/// Unicode réels.
#[inline]
fn is_emoji(c: char) -> bool {
    let u = c as u32;
    // Aucun bloc émoji ne commence en dessous de 0x2600. Ce test unique écarte
    // d'un coup les lettres, les chiffres, la ponctuation et tous les accents
    // latins — c'est-à-dire la quasi-totalité des caractères d'un tweet — au
    // lieu de leur faire subir les six comparaisons de plage suivantes. Sur un
    // vivier de 1700 tweets, ça représentait le premier poste de l'ingestion.
    if u < 0x2600 {
        return false;
    }
    (0x1F300..=0x1FAFF).contains(&u)   // emoticons + symbols & pictographs + supplemental
        || (0x2600..=0x27BF).contains(&u) // misc symbols + dingbats
        || (0x1F000..=0x1F0FF).contains(&u) // mahjong, dominoes, cards
        || (0xFE00..=0xFE0F).contains(&u)   // sélecteurs de variante (style émoji)
        || u == 0x2B50
        || u == 0x2764 // ⭐ ❤
}

/// Retient un mot s'il est assez long et que le plafond n'est pas atteint.
///
/// `fini` reste vrai une fois le plafond atteint : on cesse de retenir des
/// mots, mais le balayage continue — les compteurs de ponctuation et d'émojis
/// portaient, eux, sur le texte ENTIER.
#[inline]
fn push_word(
    spans: &mut Vec<Range<u32>>,
    start: usize,
    end: usize,
    indexable: bool,
    fini: &mut bool,
) {
    if *fini || !indexable || end - start < MIN_WORD_LEN {
        return;
    }
    spans.push(start as u32..end as u32);
    if spans.len() >= MAX_WORDS {
        *fini = true;
    }
}

/// Extrait tous les traits textuels d'un tweet en un seul passage.
pub fn analyze_content(content: &str) -> ContentFeatures {
    let lower = content.to_lowercase();

    let mut emoji_count = 0i32;
    let mut exclamation_count = 0i32;
    let mut question_count = 0i32;
    // Un mot retenu fait au moins quatre caractères plus son séparateur : le
    // texte ne peut pas en contenir plus d'un cinquième de sa longueur. Réservé
    // d'avance, le vecteur ne se réalloue plus (il partait de zéro et doublait
    // cinq fois par tweet, en recopiant à chaque fois).
    let mut word_spans: Vec<Range<u32>> =
        Vec::with_capacity((lower.len() / (MIN_WORD_LEN + 1)).min(MAX_WORDS));
    let mut word_start: Option<usize> = None;
    // Un tweet dont le texte dépasse 4 Go n'existe pas ; la borne évite que le
    // `as u32` tronque en silence si un appelant passait autre chose.
    let indexable = lower.len() <= u32::MAX as usize;

    // ⚠ Les compteurs `!` et `?` portaient sur `content`, le découpage en mots
    // et les URLs sur `lower`. On compte tout sur `lower`, la seule chaîne
    // qu'on garde : `to_lowercase` ne touche ni à la ponctuation, ni aux
    // émojis (aucun de ces blocs Unicode n'a de casse), donc les nombres sont
    // les mêmes. C'est vérifié caractère par caractère par
    // `un_passage_rend_les_memes_nombres_que_six`.
    //
    // Le découpage suit `char::is_whitespace`, exactement comme
    // `split_whitespace`, et retient les mêmes mots dans le même ordre.
    let mut fini = false;
    for (idx, c) in lower.char_indices() {
        if c.is_whitespace() {
            if let Some(start) = word_start.take() {
                push_word(&mut word_spans, start, idx, indexable, &mut fini);
            }
        } else {
            if word_start.is_none() {
                word_start = Some(idx);
            }
            match c {
                '!' => exclamation_count += 1,
                '?' => question_count += 1,
                _ if is_emoji(c) => emoji_count += 1,
                _ => {}
            }
        }
    }
    if let Some(start) = word_start {
        push_word(&mut word_spans, start, lower.len(), indexable, &mut fini);
    }

    // Compter les vraies URLs (schémas) plutôt que toute occurrence de "http".
    // `https://` contient `http` mais pas `http://` : les deux comptages ne se
    // recouvrent pas, l'addition ne double personne. Laissé en recherche de
    // sous-chaîne — `memchr` fait ça bien mieux qu'un automate écrit à la main,
    // et la plupart des tweets n'ont aucun `h` à la bonne place.
    // Une garde avant les deux recherches : l'écrasante majorité des tweets
    // n'a aucun lien, et `contains("http")` répond non en un seul balayage
    // (memchr sur le `h`) là où les deux `matches` en faisaient deux.
    let url_count = if lower.contains("http") {
        (lower.matches("http://").count() + lower.matches("https://").count()) as i32
    } else {
        0
    };

    ContentFeatures {
        lower,
        word_spans,
        emoji_count,
        exclamation_count,
        question_count,
        url_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'implémentation d'origine, recopiée telle quelle depuis `map_rows`.
    /// Elle ne sert plus qu'ici : à prouver que la version en un passage rend
    /// exactement les mêmes nombres.
    fn analyse_de_reference(content: &str) -> (i32, i32, i32, i32, Vec<String>) {
        let content_lower = content.to_lowercase();
        let emoji_count = content
            .chars()
            .filter(|c| {
                let u = *c as u32;
                (0x1F300..=0x1FAFF).contains(&u)
                    || (0x2600..=0x27BF).contains(&u)
                    || (0x1F000..=0x1F0FF).contains(&u)
                    || (0xFE00..=0xFE0F).contains(&u)
                    || u == 0x2B50
                    || u == 0x2764
            })
            .count() as i32;
        let exclamation_count = content.matches('!').count() as i32;
        let question_count = content.matches('?').count() as i32;
        let url_count = (content_lower.matches("http://").count()
            + content_lower.matches("https://").count()) as i32;
        let words: Vec<String> = content_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(String::from)
            .take(50)
            .collect();
        (
            emoji_count,
            exclamation_count,
            question_count,
            url_count,
            words,
        )
    }

    fn comparer(content: &str) {
        let f = analyze_content(content);
        let (emoji, excl, quest, urls, words) = analyse_de_reference(content);
        assert_eq!(f.emoji_count, emoji, "emojis: {content:?}");
        assert_eq!(f.exclamation_count, excl, "exclamations: {content:?}");
        assert_eq!(f.question_count, quest, "questions: {content:?}");
        assert_eq!(f.url_count, urls, "urls: {content:?}");
        assert_eq!(
            f.words().collect::<Vec<_>>(),
            words.iter().map(|w| w.as_str()).collect::<Vec<_>>(),
            "mots: {content:?}"
        );
    }

    #[test]
    fn un_passage_rend_les_memes_nombres_que_six() {
        comparer("");
        comparer("court");
        comparer("Bonjour le monde, comment allez-vous ?!");
        comparer("MAJUSCULES ET Accents ÉÀÜ ŒUF Straße");
        comparer("Trois !!! et deux ??");
        comparer("Voir http://exemple.fr et HTTPS://Exemple.fr/page");
        comparer("Emojis 😀🎉❤⭐✅ et du texte ordinaire");
        comparer("a b c d ab abc abcd abcde");
        comparer("   espaces\tet\ttabulations\nlignes   ");
        comparer("idéogrammes 漢字 qui ne sont pas des emojis");
    }

    #[test]
    fn le_plafond_de_cinquante_mots_tient() {
        let long = (0..200)
            .map(|i| format!("motnumero{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        comparer(&long);
        assert_eq!(analyze_content(&long).word_count(), MAX_WORDS);
    }

    /// Générateur déterministe — un xorshift, pas `rand` : un test qui échoue
    /// une fois sur cent sans qu'on puisse le rejouer ne sert à rien.
    struct Tirage(u64);

    impl Tirage {
        fn suivant(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn dans(&mut self, n: usize) -> usize {
            (self.suivant() % n as u64) as usize
        }
    }

    /// Le vrai filet de sécurité : mille textes tirés au hasard dans un
    /// alphabet fait exprès pour piéger l'implémentation — majuscules,
    /// accents, ligatures, ponctuation, émojis, séparateurs Unicode, mots à la
    /// limite des quatre lettres, schémas d'URL tronqués.
    #[test]
    fn mille_textes_tires_au_hasard_donnent_les_memes_nombres() {
        const MORCEAUX: &[&str] = &[
            "a", "ab", "abc", "abcd", "abcde", "Mot", "MOT", "Élan", "œuf", "Straße", "İstanbul",
            "!", "?", "!?", "😀", "❤", "⭐", "✅", "漢字", "http://x.fr", "https://X.FR/p",
            "http", "https://", "://", " ", "\t", "\n", "\u{000B}", "\u{00A0}", "\u{2003}", "",
            "trois!!!", "deux??", "a̐é", "ﬁn", "ＦＵＬＬＷＩＤＴＨ",
        ];
        let mut tirage = Tirage(0x2026_0831_C0FF_EE01);
        for _ in 0..1_000 {
            let n = tirage.dans(24);
            let mut texte = String::new();
            for _ in 0..n {
                texte.push_str(MORCEAUX[tirage.dans(MORCEAUX.len())]);
            }
            comparer(&texte);
        }
    }

    #[test]
    fn les_bornes_pointent_bien_dans_la_chaine_minuscule() {
        let f = analyze_content("Alpha BRAVO charlie");
        assert_eq!(f.words().collect::<Vec<_>>(), vec!["alpha", "bravo", "charlie"]);
    }
}
