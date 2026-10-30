"""Interlocuteur d'Aria (qui parle) : lit et fixe l'interlocuteur actif auprès du serveur de mémoire, via MCP.

Module isolé : il n'importe rien de tools/memory et ne connaît que l'outil MCP caché « ui_etat ». Sans serveur de mémoire,
toutes les fonctions se dégradent vers « Inconnue » sans erreur, et la liste déroulante de l'interface reste masquée.
"""
import json

import components.utils as utils

TOOL = "ui_etat"
INCONNUE = ("Inconnue", "")                 # (libellé, valeur) de l'entrée toujours présente de la liste
VIDE = {"choices": [INCONNUE], "actif": "", "contexte": ""}


def disponible(mcp_manager):
    """Le serveur de mémoire est-il connecté (donc la fonctionnalité utilisable) ?"""
    return bool(mcp_manager) and mcp_manager.has_tool(TOOL)


async def etat(mcp_manager, activer="?"):
    """Lit (activer='?') ou fixe l'interlocuteur actif ('' = Inconnue, sinon un identifiant) et renvoie
    {"choices": [(libellé, id)...], "actif": id, "contexte": texte}. Toute erreur renvoie l'état « Inconnue »."""
    if not disponible(mcp_manager):
        return dict(VIDE)
    try:
        raw = await mcp_manager.call_tool(TOOL, {"activer": activer if activer is not None else ""})
        data = json.loads(raw)
        choices = [INCONNUE] + [(p["label"], p["id"]) for p in data.get("liste", [])]
        actif = data.get("actif", "")
        if actif not in {value for _, value in choices}:
            actif = ""
        return {"choices": choices, "actif": actif, "contexte": data.get("contexte", "")}
    except Exception as e:
        utils.log_info("INTERLOCUTEUR", f"État indisponible, retour à Inconnue : {type(e).__name__}: {e}")
        return dict(VIDE)
