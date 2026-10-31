"""Normalisation et découpage du texte français pour la recherche (bibliothèque standard seulement).

La même fonction `analyze` sert à l'indexation et à la requête : une racinisation grossière suffit, puisque
les deux côtés produisent les mêmes racines. Module autonome : aucune importation depuis Aria.
"""
import re
import unicodedata

# Mots vides (formes sans accents). « fait » et « ete » en sont absents : ils ont un sens utile (type, saison).
STOPWORDS = frozenset("""
le la les un une des du de au aux et ou mais donc or ni car ce cet cette ces mon ma mes ton ta tes son sa ses
notre nos votre vos leur leurs je tu il elle on nous vous ils elles me te se lui en qui que quoi dont qu quel
quelle quels quelles est es sont suis etes sommes etre ai as avons avez ont avoir pas ne plus tres si sur sous
dans par pour avec sans chez vers entre comment quand pourquoi combien aussi alors ainsi cela ca ceci celui
celle ceux celles tout tous toute toutes deja encore ici bien peu trop meme autre autres tel telle fois
quelque quelques dis dit moi toi est ce oui non
utilisateur utilisatrice nom appelle appelle prenom sait savoir connais connait dire parle
""".split())

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Suffixes retirés (du plus long au plus court), à condition de garder une racine d'au moins 3 lettres
_SUFFIXES = ("ements", "ement", "ations", "ation", "euses", "euse", "eurs", "eur", "ees", "ee", "es", "ent", "er", "ir", "e")


def normalize(text):
    """Minuscules, sans accents, ligatures éclatées, tout ce qui n'est pas lettre ou chiffre devient un espace."""
    t = (text or "").lower().replace("œ", "oe").replace("æ", "ae")
    if not t.isascii():                      # texte déjà ASCII (cas fréquent) : inutile de décomposer
        t = unicodedata.normalize("NFKD", t)
        # Seules les marques d'accent tombent : les autres caractères (apostrophe typographique, €...) restent des séparateurs
        t = "".join(c for c in t if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", t).strip()


# Pluriels irréguliers en -aux / -eux : singulier correspondant (liste fermée, complétée au besoin)
_PLURIELS_IRREGULIERS = {"travaux": "travail", "vitraux": "vitrail", "coraux": "corail", "emaux": "email", "soupiraux": "soupirail",
                         "baux": "bail", "yeux": "oeil", "ciaux": "ciel", "aieux": "aieul"}
# Mots en -au / -eau dont le pluriel en -aux ne vient PAS d'un singulier en -al
_AUX_SANS_AL = frozenset(("tuyaux", "noyaux", "boyaux", "joyaux", "landaux", "fourneaux", "sarraux", "hoyaux", "etaux", "eaux"))


def _singulier(token):
    """Pluriel -> singulier pour les cas irréguliers : « animaux » -> « animal », « chevaux » -> « cheval », « travaux » -> « travail ».
    Les pluriels réguliers (-s, -x) sont traités par stem. Heuristique sans dictionnaire : un mot en -aux (hors -eaux et liste fermée
    d'exceptions) vient d'un singulier en -al."""
    if token in _PLURIELS_IRREGULIERS:
        return _PLURIELS_IRREGULIERS[token]
    if token.endswith("aux") and not token.endswith("eaux") and token not in _AUX_SANS_AL and len(token) >= 5:
        return token[:-3] + "al"
    return token


def stem(token):
    """Racinisation légère : pluriel (régulier et en -aux) puis un suffixe courant. Les nombres sont laissés tels quels."""
    if token.isdigit() or len(token) <= 3:
        return token
    token = _singulier(token)
    if token.endswith(("s", "x")):
        token = token[:-1]
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def analyze(text):
    """Texte -> liste de racines (mots vides et lettres isolées retirés ; un chiffre seul est conservé).
    L'élision (« l'école », « qu'il ») est traitée par la normalisation : l'apostrophe sépare, la lettre isolée tombe."""
    out = []
    for token in normalize(text).split():
        if token in STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        out.append(stem(token))
    return out
