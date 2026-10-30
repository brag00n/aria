"""Liens entre personnes : forme canonique, sens inverse et mots à indexer (module autonome, bibliothèque standard).

Un lien n'est enregistré qu'UNE fois. Son sens inverse (« Tiphaine est la femme d'Olivier » => « Olivier est le mari ou conjoint de
Tiphaine ») est déduit à la lecture et ajouté à l'index de recherche, jamais stocké comme second souvenir : une fratrie de cinq
personnes ne produit pas vingt liens. La forme canonique sert à reconnaître qu'on redit le même lien dans l'autre sens
(« A frère de B » puis « B sœur de A »), pour ne pas le doubler.
"""
import textproc

# mot-clé du lien (sans accents) : (famille, symétrique, phrase du sens inverse, mots du sens inverse à indexer)
_CONJOINT_F = ("conjoint", True, "le mari ou conjoint de", "mari epoux conjoint compagnon")
_CONJOINT_M = ("conjoint", True, "la femme ou conjointe de", "femme epouse conjointe compagne")
_CONJOINT_N = ("conjoint", True, "le conjoint ou la conjointe de", "conjoint conjointe mari femme epoux epouse")
_FRATRIE = ("fratrie", True, "frère ou sœur de", "frere soeur fratrie")
_PARENT = ("parent", False, "l'enfant de", "enfant fils fille")
_ENFANT = ("enfant", False, "le parent de", "parent pere mere")

LIENS = {
    "femme": _CONJOINT_F, "epouse": _CONJOINT_F,
    "mari": _CONJOINT_M, "epoux": _CONJOINT_M,
    "conjoint": _CONJOINT_N, "conjointe": _CONJOINT_N, "compagnon": _CONJOINT_N, "compagne": _CONJOINT_N,
    "frere": _FRATRIE, "soeur": _FRATRIE,
    "ami": ("ami", True, "ami(e) de", "ami amie"), "amie": ("ami", True, "ami(e) de", "ami amie"),
    "voisin": ("voisin", True, "voisin(e) de", "voisin voisine"), "voisine": ("voisin", True, "voisin(e) de", "voisin voisine"),
    "collegue": ("collegue", True, "collègue de", "collegue"),
    "pere": _PARENT, "mere": _PARENT, "parent": _PARENT,
    "fils": _ENFANT, "fille": _ENFANT, "enfant": _ENFANT,
}


# Mots qui rendent un lien composé : « belle-mère », « demi-frère », « ex-femme »... ne sont PAS « mère », « frère », « femme »
_COMPOSES = frozenset(("belle", "beau", "demi", "ex", "grand", "petit", "arriere"))


def _famille(lien):
    """Famille d'un lien connu, ou None. Seuls les liens simples sont reconnus (« femme de », « frère ou sœur de »)."""
    words = textproc.normalize(lien).split()
    if _COMPOSES.intersection(words):
        return None
    for w in words:
        if w in LIENS:
            return LIENS[w]
    return None


def analyser(a, lien, b):
    """Retourne {"cle": clé de déduplication, "invp": phrase du sens inverse ('' si inconnue), "inv": mots à indexer}.
    La clé est la même pour les deux sens d'un lien connu ; un lien inconnu garde sa forme telle quelle (sans sens inverse)."""
    ka, kb = textproc.normalize(a), textproc.normalize(b)
    famille = _famille(lien)
    if not famille:
        return {"cle": f"lien|{textproc.normalize(lien)}|{ka}|{kb}", "invp": "", "inv": ""}
    nom, symetrique, invp, inv = famille
    if symetrique:
        premier, second = sorted((ka, kb))
        cle = f"{nom}|{premier}|{second}"
    elif nom == "enfant":
        cle = f"parent|{kb}|{ka}"            # « A fils de B » = « B parent de A »
    else:
        cle = f"parent|{ka}|{kb}"
    return {"cle": cle, "invp": invp, "inv": inv}


def article(lien):
    """« femme de » -> « la femme de » ; les liens inconnus sont gardés tels quels (« est collègue de »)."""
    words = lien.split()
    first = textproc.normalize(words[0]) if words else ""
    if first in ("femme", "fille", "soeur", "mere", "amie", "voisine", "compagne"):
        return "la " + lien
    if first == "collegue":
        return "collègue " + " ".join(words[1:])
    if first in ("mari", "fils", "pere", "frere", "ami", "voisin", "compagnon", "epoux"):
        return "le " + lien
    return lien
