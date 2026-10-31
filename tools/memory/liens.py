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
    "marie": _CONJOINT_N, "mariee": _CONJOINT_N, "pacse": _CONJOINT_N, "pacsee": _CONJOINT_N,
    "frere": _FRATRIE, "soeur": _FRATRIE,
    "ami": ("ami", True, "ami(e) de", "ami amie"), "amie": ("ami", True, "ami(e) de", "ami amie"),
    "voisin": ("voisin", True, "voisin(e) de", "voisin voisine"), "voisine": ("voisin", True, "voisin(e) de", "voisin voisine"),
    "collegue": ("collegue", True, "collègue de", "collegue"),
    "pere": _PARENT, "mere": _PARENT, "parent": _PARENT,
    "fils": _ENFANT, "fille": _ENFANT, "enfant": _ENFANT,
}


# Liste FERMÉE des liens que le LLM peut enregistrer (« A est le <lien> de B »). Le sexe ne figure pas dans le lien : il vient de la fiche
# de la personne. `autre` + une précision en texte libre couvre le reste (marraine, ex-femme...) sans calcul de parenté.
TYPES = ("conjoint", "parent", "enfant", "fratrie", "demi", "beauparent", "beauenfant", "ami", "collegue", "voisin", "autre")

# phrase par défaut de chaque lien (« A est <phrase> B »)
PHRASES = {
    "conjoint": "conjoint(e) de", "parent": "parent de", "enfant": "enfant de", "fratrie": "frère ou sœur de",
    "demi": "demi-frère ou demi-sœur de", "beauparent": "beau-parent de", "beauenfant": "beau-enfant de",
    "ami": "ami(e) de", "collegue": "collègue de", "voisin": "voisin(e) de",
}

# lien déclaré (demi, beau) : (famille, symétrique, phrase du sens inverse, mots à indexer)
_DECLARES = {
    "demi": ("demi", True, "demi-frère ou demi-sœur de", "demi frere soeur"),
    "beauparent": ("beauparent", False, "le beau-enfant de", "beau fils fille enfant"),
    "beauenfant": ("beauenfant", False, "le beau-parent de", "beau pere mere parent"),
}


def choisir(lien, precision=""):
    """Traduit ce que donne le LLM en (type, texte du lien) : (None, '') si le lien n'est pas dans la liste fermée.
    - un identifiant de la liste (« conjoint », « enfant »...) : le texte est la phrase par défaut, ou `precision` si elle dit la même chose
      (« femme de » garde le sexe) ;
    - `autre` : le texte est la précision libre (obligatoire) ; si elle correspond à un lien connu, c'est ce lien qui est retenu ;
    - un mot courant de la table (« femme de », « marié à », « fille de »...) : filet de sécurité, le type est déduit du mot."""
    mot = textproc.normalize(lien).replace(" ", "")
    precision = " ".join(str(precision or "").split())[:40]
    if mot in TYPES and mot != "autre":
        texte = precision if precision and type_lien(precision) == mot else PHRASES[mot]
        return mot, texte
    if mot == "autre":
        if not precision:
            return None, ""
        connu = type_lien(precision)
        return (connu, precision) if connu else ("autre", precision)
    texte = " ".join(str(lien or "").split())[:40]
    connu = type_lien(texte)
    return (connu, texte) if connu else (None, "")


_PAR_TYPE = {"conjoint": _CONJOINT_N, "fratrie": _FRATRIE, "parent": _PARENT, "enfant": _ENFANT, "ami": LIENS["ami"],
             "voisin": LIENS["voisin"], "collegue": LIENS["collegue"], **_DECLARES}


def type_de(rel):
    """Type d'une relation stockée : son champ `t` s'il y en a un, sinon déduit du texte (anciennes relations). '' = pas de calcul."""
    t = (rel or {}).get("t", "")
    if t:
        return "" if t == "autre" else t
    return type_lien((rel or {}).get("l", ""))


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


def type_lien(lien):
    """Famille d'un lien : 'conjoint', 'parent' (A est le parent de B), 'enfant' (A est l'enfant de B), 'fratrie', 'ami'...
    ou '' pour un lien non reconnu (belle-mère, demi-frère, ex-femme... : gardés en texte libre)."""
    mots = set(textproc.normalize(lien).split())
    if "demi" in mots and mots & {"frere", "soeur"}:
        return "demi"                                  # « demi-sœur de » : déclaré, il l'emporte sur la présomption
    if mots & {"belle", "beau"}:
        if mots & {"pere", "mere", "parent"}:
            return "beauparent"                        # « A est le beau-père / la belle-mère de B »
        if mots & {"fils", "fille", "enfant"}:
            return "beauenfant"                        # « A est le beau-fils / la belle-fille de B »
        return ""                                      # beau-frère, belle-sœur : déduits des autres liens
    famille = _famille(lien)
    return famille[0] if famille else ""


_MOTS_F = frozenset(("femme", "epouse", "compagne", "mere", "fille", "soeur"))
_MOTS_M = frozenset(("mari", "epoux", "compagnon", "pere", "fils", "frere"))


def sexe_du_lien(lien):
    """Sexe de la personne qui PORTE le lien (« Tiphaine est la femme de... » -> 'f'), '' si le lien ne le dit pas
    (« conjoint », « parent », « enfant », « frère ou sœur »...)."""
    mots = set(textproc.normalize(lien).split())
    f, m = bool(mots & _MOTS_F), bool(mots & _MOTS_M)
    return "f" if f and not m else "m" if m and not f else ""


def analyser(a, lien, b, t=None):
    """Retourne {"cle": clé de déduplication, "invp": phrase du sens inverse ('' si inconnue), "inv": mots à indexer}.
    La clé est la même pour les deux sens d'un lien connu ; un lien inconnu garde sa forme telle quelle (sans sens inverse)."""
    ka, kb = textproc.normalize(a), textproc.normalize(b)
    t = t if t not in (None, "autre") else type_lien(lien) if t is None else ""
    famille = _famille(lien) or _PAR_TYPE.get(t)
    if not famille:
        return {"cle": f"lien|{textproc.normalize(lien)}|{ka}|{kb}", "invp": "", "inv": ""}
    nom, symetrique, invp, inv = famille
    if symetrique:
        premier, second = sorted((ka, kb))
        cle = f"{nom}|{premier}|{second}"
    elif nom == "enfant":
        cle = f"parent|{kb}|{ka}"            # « A fils de B » = « B parent de A »
    elif nom == "beauenfant":
        cle = f"beauparent|{kb}|{ka}"
    else:
        cle = f"{nom}|{ka}|{kb}"
    return {"cle": cle, "invp": invp, "inv": inv}


def article(lien):
    """« femme de » -> « la femme de » ; les liens inconnus sont gardés tels quels (« est collègue de »)."""
    words = lien.split()
    first = textproc.normalize(words[0]) if words else ""
    if first in ("femme", "fille", "soeur", "mere", "compagne"):
        return "la " + lien
    if first in ("mari", "fils", "pere", "frere", "compagnon", "epoux"):
        return "le " + lien
    # « ami(e) de », « voisin(e) de », « collègue de » : tournure sans article (« Paul est ami de Olivier »)
    return lien
