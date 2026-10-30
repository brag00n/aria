"""Importance, décroissance et classement des souvenirs (fonctions pures, horloge passée en paramètre).

Importance effective d'un souvenir :
    eff = imp x s x 0,5 ^ (âge_en_jours / demi-vie du type)
où `imp` est la note de départ fixée par le type, `s` le renfort par l'usage (1,0 à 2,0) et l'âge compte depuis le
dernier rappel. Les souvenirs protégés (tags core, identity, pinned) ne décroissent pas.
"""
import math

# Demi-vie en jours, par type
HALF_LIFE = {"ephemere": 10, "tache": 30, "preference": 90, "fait": 120}
# Note de départ, par type
START_IMP = {"preference": 6, "fait": 5, "tache": 4, "ephemere": 2}
PROTECTED = frozenset(("core", "identity", "pinned"))

HIDE_BELOW = 1.0        # sous ce seuil : caché des recherches
ARCHIVE_BELOW = 0.5     # sous ce seuil : archivé à la compaction (jamais supprimé)
S_MAX = 2.0             # plafond du renfort
RECALL_BOOST = 0.1      # renfort quand un souvenir est renvoyé par une recherche
DUP_BOOST = 0.25        # renfort quand le même fait est mémorisé une seconde fois
RECENT_DAYS = 7         # un souvenir rappelé depuis moins de 7 jours reçoit un léger bonus

K1 = 1.2
B = 0.75


def is_protected(tags):
    return any(t in PROTECTED for t in tags)


def effective(imp, s, last_ts, now, mtype, tags):
    """Importance effective (voir en-tête). `last_ts` et `now` sont des secondes depuis l'époque."""
    base = imp * s
    if is_protected(tags):
        return base
    age_days = max(0.0, (now - last_ts) / 86400.0)
    return base * 0.5 ** (age_days / HALF_LIFE.get(mtype, 120))


def idf(n_docs, n_with_term):
    return math.log(1 + (n_docs - n_with_term + 0.5) / (n_with_term + 0.5))


def bm25_term(tf, doc_len, avg_len):
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / max(avg_len, 1e-9)))


def final_score(bm25_norm, eff, recent):
    """bm25 normalisé (0..1) pondéré par l'importance effective et par la récence du dernier rappel."""
    return bm25_norm * (0.6 + 0.4 * min(eff, 10.0) / 10.0) * (1 + (0.1 if recent else 0.0))
