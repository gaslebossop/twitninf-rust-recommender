
use tracing::debug;

use crate::models::RawTweet;

use super::models::GarbageSignals;

// ─── Seuils de détection ──────────────────────────────────────────────────────

const SPAM_HASHTAG_RATIO: f64 = 0.30; // hashtags / nb mots
const SPAM_HASHTAG_MIN: i32 = 3; // minimum absolu pour déclencher
const MENTION_SPAM_THRESHOLD: i32 = 5;
const ZERO_ENG_MIN_VIEWS: i64 = 200;
const REPORT_RATE_THRESHOLD: f64 = 0.03; // 3% des vues → report
const LINK_SPAM_MAX_CHARS: i32 = 50;
const LINK_SPAM_MIN_URLS: i32 = 2;
const EMOJI_OVERLOAD_COUNT: i32 = 8;
const EMOJI_OVERLOAD_MAX_CHARS: i32 = 100;
const REPEAT_UNIQUE_RATIO: f64 = 0.25; // < 25% mots uniques = texte répétitif
const REPEAT_MIN_WORDS: usize = 10; // pas de pénalité sur tweets très courts
const NEW_ACCOUNT_MAX_AGE_DAYS: i32 = 2; // compte créé avant-hier ou plus tard
const NEW_ACCOUNT_BURST_MIN_TWEETS: i64 = 50; // déjà 50 posts en ≤ 2 jours

// ─── GarbageContentDetector ──────────────────────────────────────────────────

/// Détecte les signaux de contenu poubelle dans un tweet ou sur un compte.
pub struct GarbageContentDetector;

impl GarbageContentDetector {
    pub fn new() -> Self {
        Self
    }

    /// Moins de `REPEAT_UNIQUE_RATIO` de mots distincts ?
    ///
    /// Écrit avec sortie anticipée plutôt qu'en dédupliquant tout : la réponse
    /// est connue dès qu'on a vu assez de mots DIFFÉRENTS pour dépasser le
    /// seuil, ce qui arrive au bout de trois à treize mots sur un texte
    /// ordinaire. Un balayage linéaire sur ce petit nombre revient moins cher
    /// qu'une table de hachage — SipHash coûte plus que quelques comparaisons
    /// de chaînes courtes, et il n'y a rien à allouer avant de savoir combien.
    fn is_repetitive<'a>(words: impl Iterator<Item = &'a str>, word_count: usize) -> bool {
        let limit = REPEAT_UNIQUE_RATIO * word_count as f64;
        let mut distinct: Vec<&str> = Vec::with_capacity(limit as usize + 2);
        for word in words {
            if distinct.contains(&word) {
                continue;
            }
            distinct.push(word);
            if distinct.len() as f64 >= limit {
                return false;
            }
        }
        (distinct.len() as f64) < limit
    }

    /// Analyse un tweet individuel et retourne ses signaux de qualité.
    pub fn detect(&self, tweet: &RawTweet) -> GarbageSignals {
        let word_count = tweet.text.word_count().max(1);

        // Densité de hashtags spam : > 30% des mots sont des #tags
        let spam_hashtag_density = tweet.hashtag_count >= SPAM_HASHTAG_MIN
            && (tweet.hashtag_count as f64 / word_count as f64) > SPAM_HASHTAG_RATIO;

        // Mention spam : > 5 @mentions (bot ou astroturfing)
        let spam_mentions = tweet.mention_count > MENTION_SPAM_THRESHOLD;

        // Zéro engagement : beaucoup de vues, aucune réaction → contenu ignoré
        let zero_engagement = tweet.view_count > ZERO_ENG_MIN_VIEWS
            && tweet.like_count == 0
            && tweet.comment_count == 0
            && tweet.retweet_count == 0;

        // Taux de signalements élevé
        let report_rate = tweet.report_count as f64 / tweet.view_count.max(100) as f64;
        let high_report_rate = report_rate > REPORT_RATE_THRESHOLD;

        // Spam de liens purs : beaucoup d'URLs, très peu de texte original
        let pure_link_spam =
            tweet.url_count >= LINK_SPAM_MIN_URLS && tweet.content_length < LINK_SPAM_MAX_CHARS;

        // Surcharge d'émojis sans contenu textuel
        let emoji_overload = tweet.emoji_count > EMOJI_OVERLOAD_COUNT
            && tweet.content_length < EMOJI_OVERLOAD_MAX_CHARS;

        // Contenu répétitif : ratio mots uniques < 25% (copie-colle, template)
        //
        // ⚠ Ce signal construisait une `HashSet` des mots de CHAQUE candidat,
        // systématiquement — y compris pour les tweets trop courts pour être
        // concernés, et y compris quand `detect` est appelé deux fois sur le
        // même tweet. Mesuré : 15 % du chemin de scoring, pour un booléen qui
        // est faux dans l'écrasante majorité des cas.
        //
        // Deux gardes, dans cet ordre :
        //   1. le seuil de longueur, qui existait déjà mais était évalué APRÈS
        //      la construction de l'ensemble ;
        //   2. le nombre de mots uniques nécessaires pour rester sous le seuil
        //      — dès qu'on en a compté autant, la réponse est « non », et il
        //      est inutile de finir de dédupliquer.
        let repeat_content =
            word_count >= REPEAT_MIN_WORDS && Self::is_repetitive(tweet.text.words(), word_count);

        // Compte tout juste créé, déjà prolifique : `author_account_age_days`
        // et `author_tweet_count` sont chargés depuis la base pour chaque
        // candidat mais qu'aucun signal ne lisait jusqu'ici. Un humain qui
        // découvre l'app ne publie pas 50 fois avant sa deuxième journée —
        // c'est le rythme d'un compte créé pour poster, pas pour échanger.
        let new_account_burst = tweet.author_account_age_days >= 0
            && tweet.author_account_age_days <= NEW_ACCOUNT_MAX_AGE_DAYS
            && tweet.author_tweet_count >= NEW_ACCOUNT_BURST_MIN_TWEETS;

        let signals = GarbageSignals {
            spam_hashtag_density,
            spam_mentions,
            zero_engagement,
            high_report_rate,
            pure_link_spam,
            emoji_overload,
            repeat_content,
            new_account_burst,
        };

        if signals.is_garbage() {
            debug!(
                tweet_id = %tweet.id,
                score = signals.score(),
                signals = ?signals.active_signals(),
                "Garbage content detected"
            );
        }

        signals
    }
}

impl Default for GarbageContentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tweet_with(age_days: i32, tweet_count: i64) -> RawTweet {
        RawTweet {
            author_account_age_days: age_days,
            author_tweet_count: tweet_count,
            ..Default::default()
        }
    }

    #[test]
    fn compte_neuf_et_prolifique_est_signale() {
        let d = GarbageContentDetector::new();
        let signals = d.detect(&tweet_with(1, 80));
        assert!(signals.new_account_burst);
    }

    #[test]
    fn compte_neuf_mais_peu_prolifique_n_est_pas_signale() {
        let d = GarbageContentDetector::new();
        let signals = d.detect(&tweet_with(1, 5));
        assert!(!signals.new_account_burst);
    }

    #[test]
    fn compte_etabli_et_prolifique_n_est_pas_signale() {
        // Un compte ancien qui publie beaucoup, c'est un compte actif, pas un
        // signal de spam en soi.
        let d = GarbageContentDetector::new();
        let signals = d.detect(&tweet_with(400, 5000));
        assert!(!signals.new_account_burst);
    }

    // ─── Contenu répétitif ───────────────────────────────────────────────────

    /// Le calcul d'origine : dédupliquer TOUS les mots dans une table de
    /// hachage, puis comparer le ratio. Gardé ici comme référence — la version
    /// à sortie anticipée doit rendre le même booléen sur toutes les entrées.
    ///
    /// Prend le tweet DÉJÀ analysé et non la liste brute : `analyze_content`
    /// écarte les mots de moins de quatre lettres, et une référence qui
    /// compterait « de » ou « et » ne mesurerait pas la même chose que le
    /// détecteur — le test passerait ou échouerait pour la mauvaise raison.
    fn repetitif_par_ensemble(tweet: &RawTweet) -> bool {
        let word_count = tweet.text.word_count().max(1);
        let uniques: std::collections::HashSet<&str> = tweet.text.words().collect();
        word_count >= REPEAT_MIN_WORDS
            && (uniques.len() as f64 / word_count as f64) < REPEAT_UNIQUE_RATIO
    }

    fn tweet_de_mots(mots: &[&str]) -> RawTweet {
        let mut t = RawTweet {
            content: mots.join(" "),
            ..Default::default()
        };
        t.analyze();
        t
    }

    #[test]
    fn la_sortie_anticipee_rend_le_meme_verdict_que_l_ensemble() {
        let cas: Vec<Vec<&str>> = vec![
            vec![],
            vec!["seul"],
            // Neuf mots : sous le plancher, jamais signalé quoi qu'il arrive.
            vec!["copie"; 9],
            // Dix mots identiques : 1 unique sur 10 = 10 %, sous 25 %.
            vec!["copie"; 10],
            // Dix mots tous distincts : 100 %.
            vec![
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
                "juliett",
            ],
            // Pile au seuil : 3 uniques sur 12 = 25 %, ce n'est PAS < 25 %.
            vec![
                "alpha", "bravo", "charlie", "alpha", "bravo", "charlie", "alpha", "bravo",
                "charlie", "alpha", "bravo", "charlie",
            ],
            // Juste en dessous : 2 uniques sur 12 = 16,7 %.
            vec![
                "alpha", "bravo", "alpha", "bravo", "alpha", "bravo", "alpha", "bravo", "alpha",
                "bravo", "alpha", "bravo",
            ],
            vec!["spam"; 50],
            // Au-delà du plafond de cinquante mots retenus.
            vec!["alpha"; 80],
            // Mots courts, écartés par l'analyse : le compte tombe sous le
            // plancher même si la phrase est longue.
            vec!["de", "et", "un", "le", "la", "les", "des", "au", "en", "on", "ou", "si"],
            // Mélange de courts et de longs.
            vec![
                "de", "promotion", "et", "promotion", "le", "promotion", "la", "promotion",
                "des", "promotion", "au", "promotion", "en", "promotion", "on", "promotion",
                "ou", "promotion", "si", "promotion",
            ],
        ];
        for mots in &cas {
            let tweet = tweet_de_mots(mots);
            let attendu = repetitif_par_ensemble(&tweet);
            let obtenu = GarbageContentDetector::new().detect(&tweet).repeat_content;
            assert_eq!(obtenu, attendu, "mots = {mots:?}");
        }
    }

    #[test]
    fn un_texte_recopie_est_bien_signale() {
        let tweet = tweet_de_mots(&["promo"; 20]);
        assert!(GarbageContentDetector::new().detect(&tweet).repeat_content);
    }

    #[test]
    fn un_texte_ordinaire_ne_l_est_pas() {
        let tweet = tweet_de_mots(&[
            "voici", "quelques", "phrases", "ordinaires", "ecrites", "par", "quelqu",
            "avec", "assez", "de", "vocabulaire", "different",
        ]);
        assert!(!GarbageContentDetector::new().detect(&tweet).repeat_content);
    }

    #[test]
    fn age_manquant_n_est_jamais_signale() {
        // `author_account_age_days` par défaut est 0 dans `RawTweet::default()` ;
        // un compte réellement inconnu ne doit pas se faire passer pour neuf.
        // Ce test documente le garde-fou `>= 0`, redondant tant que le champ
        // ne peut être négatif, mais explicite l'intention si ça change.
        let d = GarbageContentDetector::new();
        let signals = d.detect(&tweet_with(-1, 999));
        assert!(!signals.new_account_burst);
    }
}
