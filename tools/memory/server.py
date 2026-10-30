"""Serveur MCP de mémoire à long terme d'Aria (stdio). Module autonome : aucune importation depuis Aria.

Trois outils seulement (chaque outil de plus coûte du temps de prompt aux petits modèles) :
  memoriser(fait, type, tags)      retient une information durable
  rechercher_souvenir(question)    retrouve des souvenirs (question vide = les plus importants)
  oublier(id)                      oublie un souvenir (archivé, jamais perdu en silence)

Variables d'environnement :
  MEMORY_DIR         dossier des données (défaut : <ce dossier>/../../models/memory)
  MEMORY_NAMESPACE   espace de noms (défaut : default)
  MEMORY_UI_PORT     port de la page de gestion (absent = pas de page)
  MEMORY_UI_TOKEN    jeton optionnel exigé par la page
  MEMORY_LOG_TEXT    0 = ne pas consigner le texte des requêtes dans le journal d'activité (empreinte à la place)
  MEMORY_LOG_LEVEL   info (défaut) | debug
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # modules frères (store, textproc...) : aucun paquet à installer

from mcp.server.fastmcp import FastMCP  # noqa: E402

import store as store_mod  # noqa: E402
from activity import Activity  # noqa: E402

DATA_DIR = Path(os.environ.get("MEMORY_DIR") or (HERE.parent.parent / "models" / "memory"))
NAMESPACE = os.environ.get("MEMORY_NAMESPACE", "default")

activity = Activity(DATA_DIR / "logs",
                    log_text=os.environ.get("MEMORY_LOG_TEXT", "1") != "0",
                    level=os.environ.get("MEMORY_LOG_LEVEL", "info"))
store = store_mod.Store(DATA_DIR, NAMESPACE, activity=activity)

mcp = FastMCP("memoire")


def _line(rec):
    """Une ligne compacte par souvenir, c'est tout ce que le LLM reçoit."""
    return f"{rec['id']} [{rec['t']}] {rec['txt']}"


@mcp.tool(name="memoriser")
def memoriser(fait: str, type: str = "fait", tags: str = "") -> str:
    """Retient une information durable sur l'utilisateur ou sa maison. type: fait, preference, tache ou ephemere. tags: mots séparés par des virgules."""
    status, rec = store.add(fait, type, tags, src="llm")
    if status == "added":
        return f"OK {rec['id']}"
    if status == "duplicate":
        return f"Déjà connu ({rec['id']}), renforcé."
    if status == "locked":
        return "Mémoire verrouillée, rien n'a été enregistré."
    if status == "rate_limit":
        return "Trop d'enregistrements d'affilée, réessaie dans une minute."
    return "Rien à retenir."


@mcp.tool(name="rechercher_souvenir")
def rechercher_souvenir(question: str = "", limite: int = 5) -> str:
    """Cherche dans la mémoire ce que l'on sait de l'utilisateur. À appeler avant de répondre sur ses goûts, sa maison ou ses affaires."""
    results, fallback = store.search(question, limite, src="llm")
    if not results:
        return "Aucun souvenir."
    lines = [_line(r) for r in results]
    if fallback:
        lines.insert(0, "Aucune correspondance exacte, souvenirs principaux :")
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
