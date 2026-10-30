"""Serveur MCP de mémoire à long terme d'Aria (stdio). Module autonome : aucune importation depuis Aria.

Cinq outils pour le LLM (chaque outil de plus coûte du temps de prompt aux petits modèles) :
  memoriser(fait, type, tags)                     retient une information durable de l'interlocuteur actif
  rechercher_souvenir(question)                   retrouve ses souvenirs et ceux du foyer (question vide = les plus importants)
  oublier(id)                                     oublie un souvenir (archivé, jamais perdu en silence)
  identifier_interlocuteur(nom, complement)       désigne (ou crée) la personne qui parle
  relier(personne_a, lien, personne_b)            mémorise un lien entre deux personnes
Un outil caché au LLM (préfixe ui_), pour Aria seule : ui_etat(activer) -> interlocuteurs, actif et consigne du tour.

Variables d'environnement :
  MEMORY_DIR         dossier des données (défaut : <ce dossier>/../../models/memory)
  MEMORY_NAMESPACE   espace de noms (défaut : default)
  MEMORY_UI_PORT     port de la page de gestion (absent = pas de page)
  MEMORY_UI_TOKEN    jeton optionnel exigé par la page
  MEMORY_LOG_TEXT    0 = ne pas consigner le texte des requêtes dans le journal d'activité (empreinte à la place)
  MEMORY_LOG_LEVEL   info (défaut) | debug
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # modules frères (store, textproc...) : aucun paquet à installer

from mcp.server.fastmcp import FastMCP  # noqa: E402

import store as store_mod  # noqa: E402
from activity import Activity  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402

DATA_DIR = Path(os.environ.get("MEMORY_DIR") or (HERE.parent.parent / "models" / "memory"))
NAMESPACE = os.environ.get("MEMORY_NAMESPACE", "default")

activity = Activity(DATA_DIR / "logs",
                    log_text=os.environ.get("MEMORY_LOG_TEXT", "1") != "0",
                    level=os.environ.get("MEMORY_LOG_LEVEL", "info"))
store = store_mod.Store(DATA_DIR, NAMESPACE, activity=activity)
interlocuteurs = Interlocuteurs(store)

mcp = FastMCP("memoire")


def _line(rec):
    """Une ligne compacte par souvenir, c'est tout ce que le LLM reçoit."""
    return f"{rec['id']} [{rec['t']}] {rec['txt']}"


@mcp.tool(name="identifier_interlocuteur")
def identifier_interlocuteur(nom: str, complement: str = "", surnom: str = "") -> str:
    """Désigne la personne qui parle: nom = son prénom, complement = nom de famille (distingue les homonymes), surnom = comment elle veut être appelée. La crée si inconnue."""
    return interlocuteurs.identifier(nom, complement, surnom)


@mcp.tool(name="relier")
def relier(personne_a: str, lien: str, personne_b: str) -> str:
    """Mémorise un lien entre deux personnes, par exemple Tiphaine, femme de, Olivier. « moi » désigne l'interlocuteur actuel."""
    return interlocuteurs.relier(personne_a, lien, personne_b)


@mcp.tool(name="ui_etat")
def ui_etat(activer: str = "?") -> str:
    """Réservé à l'interface d'Aria (non proposé au LLM). Renvoie en JSON les interlocuteurs, l'actif et la consigne du tour. activer: '?' ne change rien, '' = Inconnue, sinon l'identifiant de la personne."""
    return json.dumps(interlocuteurs.etat(activer), ensure_ascii=False)


@mcp.tool(name="memoriser")
def memoriser(fait: str, type: str = "fait", tags: str = "") -> str:
    """Retient une information durable de l'interlocuteur actuel. type: fait, preference, tache, ephemere, ou foyer (concerne toute la maison). tags: mots séparés par des virgules."""
    status, rec = store.add(fait, type, tags, src="llm")
    if status == "added":
        return f"OK {rec['id']}"
    if status == "duplicate":
        return f"Déjà connu ({rec['id']}), renforcé."
    return interlocuteurs.refus(status)


@mcp.tool(name="rechercher_souvenir")
def rechercher_souvenir(question: str = "", limite: int = 5) -> str:
    """Cherche ce que l'on sait de l'interlocuteur actuel et du foyer. À appeler avant de répondre sur ses goûts, sa maison, sa famille ou ses affaires."""
    results, fallback = store.search(question, limite, src="llm")
    # Liens de la famille : permettent de répondre sur un lien indirect (« qui suis-je pour Tiphaine ? ») sans deuxième recherche
    proches = interlocuteurs.entourage(question, deja={r["id"] for r in results})
    calcules = interlocuteurs.liens_calcules(question)
    if not results and not proches and not calcules:
        return "Aucun souvenir."
    lines = []
    if calcules:
        # Liens déjà déduits par le moteur de parenté : à reprendre tels quels, sans les réinterpréter
        lines.append("Parenté établie (à utiliser telle quelle) :")
        lines.extend(f"- {phrase}" for phrase in calcules)
    if fallback and results:
        lines.append("Aucune correspondance exacte, souvenirs principaux :")
    lines.extend(_line(r) for r in results)
    if proches:
        lines.append("Liens connus dans la famille :")
        lines.extend(f"- {r['txt']}" for r in proches)
    return "\n".join(lines)


@mcp.tool(name="oublier")
def oublier(id: str) -> str:
    """Oublie un souvenir à partir de son identifiant (donné par rechercher_souvenir)."""
    return "Oublié." if store.delete((id or "").strip(), why="user", src="llm") else "Souvenir introuvable."


if __name__ == "__main__":
    port = os.environ.get("MEMORY_UI_PORT")
    if port:
        try:
            import ui
            ui.start(store, activity, int(port), os.environ.get("MEMORY_UI_TOKEN") or None)
        except Exception as e:     # la page est optionnelle : une erreur ne doit jamais empêcher le serveur de démarrer
            activity.log("error", where="ui_start", msg=f"{type(e).__name__}: {e}")
            activity.warn(f"page de gestion non démarrée : {e}")
    mcp.run(transport="stdio")
