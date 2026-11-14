"""Trace d'audit d'un souvenir ou d'une personne : l'échange qui l'a fait naître. Module autonome (bibliothèque standard).

Champ `tr` (absent des anciennes données : aucune migration) :
    {"e":"e_…","ts":"ISO UTC","src":"aria" | "k_2b7cfa:Robot","who":"…","x":"extrait de la question, 200 caractères au plus"}

L'échange arrive dans le `_meta` du message MCP, posé par Aria (components/mcp.py) HORS des arguments de l'outil : le LLM ne le voit pas et ne peut ni le
fournir ni le modifier. Tout est revalidé ici (format, longueurs) ; une valeur invalide donne `None` (pas de trace), jamais une erreur.
"""
import re

EXCHANGE_ID_RE = re.compile(r"^e_[0-9a-z]{6,40}$")
LIMITS = {"ts": 40, "src": 80, "who": 60, "x": 200}


def clean(meta):
    """`_meta` reçu (clés echange, ts, src, who, x) ou `tr` déjà stocké (clés e, ts, ...) -> `tr` normalisé, ou None."""
    if not isinstance(meta, dict):
        return None
    e = meta.get("echange", meta.get("e"))
    if not isinstance(e, str) or not EXCHANGE_ID_RE.match(e):
        return None
    out = {"e": e}
    for field, limit in LIMITS.items():
        value = meta.get(field, "")
        out[field] = " ".join(str(value if value is not None else "").split())[:limit]
    return out


def from_context(ctx):
    """Trace de l'appel d'outil en cours, lue dans `_meta` (ctx = Context de FastMCP, ou None)."""
    try:
        meta = ctx.request_context.meta
        if meta is None:
            return None
        extra = getattr(meta, "model_extra", None)
        return clean(extra if isinstance(extra, dict) else None)
    except Exception:
        return None
