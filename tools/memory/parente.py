"""Moteur de parenté : à partir des liens enregistrés (parent, enfant, conjoint, fratrie), nomme le lien entre deux personnes.

Les LLM se trompent sur ces déductions (« belle-fille » pour une fille). Ici elles sont calculées : on construit un graphe
(qui est parent de qui, qui est conjoint de qui), puis on cherche le chemin le plus court entre deux personnes et on le traduit en
français (grand-mère, oncle, demi-frère, belle-fille, cousin...). Le vocabulaire est inspiré de celui du projet Gramps
(relationship.py, rel_fr.py) ; aucun code n'en est repris. Module autonome, bibliothèque standard seulement.

Principe, comme en généalogie : on remonte de A jusqu'à un ancêtre commun (Ga générations), puis on redescend jusqu'à B (Gb
générations). (1, 0) = parent ; (0, 1) = enfant ; (1, 1) = frère ou sœur ; (2, 0) = grand-parent ; (1, 2) = nièce ou neveu...
Le sexe vient des personnes (champ `sexe`) ; inconnu, la formulation est neutre (« parent », « enfant »).
"""
import textproc

# ---------------------------------------------------------------------------------------------------- graphe

def _cle(nom):
    return textproc.normalize(nom)


class Famille:
    """Graphe familial. Les nœuds sont des identités (id de personne, ou prénom normalisé si la personne n'est pas connue)."""

    def __init__(self):
        self.parents = {}      # enfant -> {parents}
        self.conjoints = {}    # personne -> {conjoints}
        self.fratrie = {}      # personne -> {frères/sœurs déclarés sans parent commun connu}
        self.sexe = {}         # identité -> 'f' | 'm'
        self.noms = {}         # identité -> libellé affichable

    def _nom(self, ident, label):
        self.noms.setdefault(ident, label)

    def ajouter_parent(self, parent, enfant):
        self.parents.setdefault(enfant, set()).add(parent)

    def ajouter_conjoint(self, a, b):
        self.conjoints.setdefault(a, set()).add(b)
        self.conjoints.setdefault(b, set()).add(a)

    def ajouter_fratrie(self, a, b):
        self.fratrie.setdefault(a, set()).add(b)
        self.fratrie.setdefault(b, set()).add(a)

    def enfants(self, parent):
        return {e for e, ps in self.parents.items() if parent in ps}


def construire(relations, resoudre, sexe_de):
    """Construit la famille à partir des relations stockées.
    relations : liste de dicts {"a","l","b","type"} (type = 'parent', 'enfant', 'conjoint', 'fratrie' ou autre)
    resoudre(nom) -> identité (voir interlocuteur._ident) ; sexe_de(identité) -> 'f' | 'm' | ''."""
    fam = Famille()
    for rel in relations:
        a, b = resoudre(rel["a"]), resoudre(rel["b"])
        if not a or not b or a == b:
            continue
        fam._nom(a, rel["a"])
        fam._nom(b, rel["b"])
        t = rel.get("type", "")
        if t == "parent":            # « A est le parent de B »
            fam.ajouter_parent(a, b)
        elif t == "enfant":          # « A est l'enfant de B »
            fam.ajouter_parent(b, a)
        elif t == "conjoint":
            fam.ajouter_conjoint(a, b)
        elif t == "fratrie":
            fam.ajouter_fratrie(a, b)
        for ident in (a, b):
            s = sexe_de(ident)
            if s:
                fam.sexe[ident] = s
    return fam


# ---------------------------------------------------------------------------------------------------- chemins

def _ancetres(fam, personne, limite=6):
    """{ancêtre: génération} pour la personne elle-même (0) et ses ancêtres, en largeur d'abord (le plus proche d'abord)."""
    vus = {personne: 0}
    front = [personne]
    for gen in range(1, limite + 1):
        suivant = []
        for p in front:
            for parent in fam.parents.get(p, ()):
                if parent not in vus:
                    vus[parent] = gen
                    suivant.append(parent)
        front = suivant
        if not front:
            break
    return vus


def _parente_sang(fam, a, b):
    """Meilleur lien de sang de a vers b : (Ga, Gb, ancetres_communs) avec Ga = générations de a à l'ancêtre commun, Gb = de b.
    None si aucun ancêtre commun connu. 'Meilleur' = le plus proche (Ga + Gb minimal)."""
    aa, ab = _ancetres(fam, a), _ancetres(fam, b)
    communs = set(aa) & set(ab)
    if not communs:
        return None
    meilleur = min(aa[c] + ab[c] for c in communs)
    proches = [c for c in communs if aa[c] + ab[c] == meilleur]
    ga = min(aa[c] for c in proches)
    gb = ab[[c for c in proches if aa[c] == ga][0]]
    return ga, gb, proches


# ---------------------------------------------------------------------------------------------------- vocabulaire

def _g(sexe, masc, fem, neutre):
    return fem if sexe == "f" else masc if sexe == "m" else neutre


_ORDINAL = {1: "premier", 2: "deuxième", 3: "troisième", 4: "quatrième", 5: "cinquième"}


def _ascendant(g, sexe):
    """Génération g >= 1 au-dessus : parent, grand-parent, arrière-grand-parent..."""
    if g == 1:
        return _g(sexe, "le père", "la mère", "un parent")
    if g == 2:
        return _g(sexe, "le grand-père", "la grand-mère", "un grand-parent")
    if g == 3:
        return _g(sexe, "l'arrière-grand-père", "l'arrière-grand-mère", "un arrière-grand-parent")
    return _g(sexe, "un ascendant lointain", "une ascendante lointaine", "un ascendant lointain")


def _descendant(g, sexe):
    if g == 1:
        return _g(sexe, "le fils", "la fille", "un enfant")
    if g == 2:
        return _g(sexe, "le petit-fils", "la petite-fille", "un petit-enfant")
    if g == 3:
        return _g(sexe, "l'arrière-petit-fils", "l'arrière-petite-fille", "un arrière-petit-enfant")
    return _g(sexe, "un descendant lointain", "une descendante lointaine", "un descendant lointain")


def _collateral_haut(ga, gb, sexe):
    """a est plus proche de l'ancêtre commun que b ne l'est de lui... (ga < gb) => a est l'oncle/tante de b (gb - ga = 1)."""
    ecart = gb - ga
    if ga == 1 and ecart == 1:
        return _g(sexe, "l'oncle", "la tante", "un oncle ou une tante")
    if ga == 1 and ecart == 2:
        return _g(sexe, "le grand-oncle", "la grand-tante", "un grand-oncle ou une grand-tante")
    return _g(sexe, "un parent éloigné", "une parente éloignée", "un parent éloigné")


def _collateral_bas(ga, gb, sexe):
    """gb < ga : a est le neveu/nièce de b."""
    ecart = ga - gb
    if gb == 1 and ecart == 1:
        return _g(sexe, "le neveu", "la nièce", "un neveu ou une nièce")
    if gb == 1 and ecart == 2:
        return _g(sexe, "le petit-neveu", "la petite-nièce", "un petit-neveu ou une petite-nièce")
    return _g(sexe, "un parent éloigné", "une parente éloignée", "un parent éloigné")


def _cousin(ga, sexe):
    """ga == gb >= 2 : cousins. ga=2 : cousin germain ; ga=3 : issu de germain ; au-delà : au Nème degré."""
    nom = _g(sexe, "cousin", "cousine", "cousin ou cousine")
    if ga == 2:
        return {"m": "le cousin germain", "f": "la cousine germaine"}.get(sexe, "un cousin germain ou une cousine germaine")
    if ga == 3:
        return f"le {nom} issu de germain" if sexe == "m" else f"la {nom} issue de germain" if sexe == "f" else "un cousin issu de germain"
    return f"un {nom} éloigné" if sexe != "f" else f"une {nom} éloignée"


def _libelle_sang(ga, gb, sexe_b):
    """Ce qu'est b POUR a (b est le ... de a)... Convention : on décrit b par rapport à a, avec le sexe de b."""
    if ga == 0 and gb == 0:
        return "la même personne"
    if gb == 0:
        return _ascendant(ga, sexe_b)               # b est un ancêtre de a
    if ga == 0:
        return _descendant(gb, sexe_b)              # b est un descendant de a
    if ga == 1 and gb == 1:
        return _g(sexe_b, "le frère", "la sœur", "un frère ou une sœur")
    if ga == gb:
        return _cousin(ga, sexe_b)
    if ga < gb:                                     # b est plus bas dans la lignée : b est le neveu/nièce de a
        return _collateral_bas(gb, ga, sexe_b)
    return _collateral_haut(gb, ga, sexe_b)         # b est plus haut : b est l'oncle/tante de a


# ---------------------------------------------------------------------------------------------------- lien entre deux personnes

def lien(fam, a, b):
    """Nom du lien de b par rapport à a (« b est ... de a »), ou '' si aucun lien connu. Essaie dans l'ordre :
    1. conjoint direct ; 2. lien de sang ; 3. demi-fratrie ; 4. alliance (belle-famille) via un conjoint ; 5. fratrie déclarée."""
    if a == b:
        return "la même personne"
    sb = fam.sexe.get(b, "")

    if b in fam.conjoints.get(a, ()):
        return _g(sb, "le mari", "la femme", "le conjoint ou la conjointe")

    sang = _parente_sang(fam, a, b)
    if sang:
        ga, gb, communs = sang
        if ga == 1 and gb == 1:
            # frère/sœur de sang : mêmes deux parents, sinon demi-fratrie
            if len(communs) < 2 and (len(fam.parents.get(a, ())) > 1 or len(fam.parents.get(b, ())) > 1):
                return _g(sb, "le demi-frère", "la demi-sœur", "un demi-frère ou une demi-sœur")
        return _libelle_sang(ga, gb, sb)

    if b in fam.fratrie.get(a, ()):
        return _g(sb, "le frère", "la sœur", "un frère ou une sœur")

    # Alliance : b est lié à a par le conjoint de l'un des deux, ou b est le conjoint d'un proche de a
    for conj in fam.conjoints.get(a, ()):                       # parenté de b avec le conjoint de a
        sang = _parente_sang(fam, conj, b)
        if sang:
            return _alliance_via_conjoint_de_a(sang, sb)
    for conj in fam.conjoints.get(b, ()):                       # b est le conjoint d'un proche de a
        sang = _parente_sang(fam, a, conj)
        if sang:
            return _alliance_conjoint_de_proche(sang, sb)
    return ""


def _alliance_via_conjoint_de_a(sang, sb):
    """b est un parent de sang du conjoint de a : beau-père, belle-mère, beau-frère, belle-sœur, beau-fils..."""
    ga, gb, _ = sang
    if gb == 0:                                  # b est un ancêtre du conjoint
        if ga == 1:
            return _g(sb, "le beau-père", "la belle-mère", "un beau-parent")
        return _g(sb, "un grand-parent par alliance", "une grand-mère par alliance", "un grand-parent par alliance")
    if ga == 0:                                  # b est un descendant du conjoint : beau-fils / belle-fille
        if gb == 1:
            return _g(sb, "le beau-fils", "la belle-fille", "un beau-fils ou une belle-fille")
        return _g(sb, "un petit-fils par alliance", "une petite-fille par alliance", "un petit-enfant par alliance")
    if ga == 1 and gb == 1:
        return _g(sb, "le beau-frère", "la belle-sœur", "un beau-frère ou une belle-sœur")
    return _g(sb, "un proche par alliance", "une proche par alliance", "un proche par alliance")


def _alliance_conjoint_de_proche(sang, sb):
    """b est le conjoint d'un proche de sang de a : gendre, bru, beau-frère, belle-sœur, oncle ou tante par alliance..."""
    ga, gb, _ = sang                              # lien de a vers le proche
    if ga == 0 and gb == 1:                       # le proche est un enfant de a => b est son gendre ou sa bru
        return _g(sb, "le gendre", "la bru", "le gendre ou la bru")
    if ga == 1 and gb == 0:                       # le proche est un parent de a => b est le beau-parent
        return _g(sb, "le beau-père", "la belle-mère", "un beau-parent")
    if ga == 1 and gb == 1:                       # le proche est un frère/sœur de a
        return _g(sb, "le beau-frère", "la belle-sœur", "un beau-frère ou une belle-sœur")
    if ga == 1 and gb == 2:                       # le proche est un neveu/nièce de a
        return _g(sb, "le mari de sa nièce ou de son neveu", "la femme de sa nièce ou de son neveu", "le conjoint d'un neveu ou d'une nièce")
    if ga == 2 and gb == 1:                       # le proche est un oncle/tante de a
        return _g(sb, "l'oncle par alliance", "la tante par alliance", "un oncle ou une tante par alliance")
    return _g(sb, "un proche par alliance", "une proche par alliance", "un proche par alliance")


def phrase(fam, a, b, nom_a, nom_b):
    """« Tiphaine est la mère de Marion. » ; '' si aucun lien connu. nom_a, nom_b : libellés affichés."""
    nom_lien = lien(fam, a, b)
    if not nom_lien or nom_lien == "la même personne":
        return ""
    # nom_lien décrit b par rapport à a : « b est le/la ... de a »
    return f"{nom_b} est {nom_lien} de {nom_a}"


def tous_les_liens(fam, a, personnes):
    """{b: libellé} pour tous les b de `personnes` ayant un lien connu avec a (a exclu)."""
    resultats = {}
    for b in personnes:
        if b == a:
            continue
        nom_lien = lien(fam, a, b)
        if nom_lien:
            resultats[b] = nom_lien
    return resultats
