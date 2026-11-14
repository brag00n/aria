"""Serveur MCP de mémoire à long terme d'Aria (stdio). Module autonome : aucune importation depuis Aria.

Cinq outils pour le LLM (chaque outil de plus coûte du temps de prompt aux petits modèles) :
  memoriser(fait, type, tags)                     retient une information durable de l'interlocuteur actif
  rechercher_souvenir(question)                   retrouve ses souvenirs et ceux du foyer (question vide = les plus importants)
  oublier(id)                                     oublie un souvenir (archivé, jamais perdu en silence)
  identifier_interlocuteur(nom, complement)       désigne (ou crée) la personne qui parle
  relier(personne_a, lien, personne_b, precision) mémorise « a est le <lien> de b » (liste fermée de liens)
Un outil caché au LLM (préfixe ui_), pour Aria seule : ui_etat(activer) -> interlocuteurs, actif et consigne du tour.

Variables d'environnement :
  MEMORY_DIR         dossier des données (défaut : <ce dossier>/../../models/memory)
  MEMORY_NAMESPACE   espace de noms (défaut : default)
  MEMORY_UI_PORT     port de la page de gestion (absent = pas de page)
  MEMORY_UI_TOKEN    jeton optionnel exigé par la page
  MEMORY_HTTP_PORT   port du canal HTTP des accès tiers (127.0.0.1 seulement ; absent = pas de canal)
  MEMORY_HTTP_SECRET secret exigé par ce canal (≥ 16 caractères) ; sans lui le canal n'est pas ouvert
  MEMORY_LOG_TEXT    0 = ne pas consigner le texte des requêtes dans le journal d'activité (empreinte à la place)
  MEMORY_LOG_LEVEL   info (défaut) | debug
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # modules frères (store, textproc...) : aucun paquet à installer

from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

import store as store_mod  # noqa: E402
import audit_trace as trace_mod  # noqa: E402
from activity import Activity  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402

DATA_DIR = Path(os.environ.get("MEMORY_DIR") or (HERE.parent.parent / "models" / "memory"))
NAMESPACE = os.environ.get("MEMORY_NAMESPACE", "default")

activity = Activity(DATA_DIR / "logs",
                    log_text=os.environ.get("MEMORY_LOG_TEXT", "1") != "0",
                    level=os.environ.get("MEMORY_LOG_LEVEL", "info"))
store = store_mod.Store(DATA_DIR, NAMESPACE, activity=activity)
interlocuteurs = Interlocuteurs(store)
service = Service(store, interlocuteurs)          # cœur des outils : partagé par la façade stdio (Aria) et la façade HTTP (accès tiers)
if not store.read_only:
    try:
        interlocuteurs.rattraper()          # fiches et sexes des personnes citées dans d'anciens liens (idempotent)
        interlocuteurs.migrer()             # type de chaque ancien lien (liste fermée), identifiants de personnes (idempotent)
    except Exception as e:                  # jamais bloquant : le serveur doit démarrer
        activity.warn(f"rattrapage des fiches impossible : {e}")

mcp = FastMCP("memoire")


def _line(rec):
    """Une ligne compacte par souvenir, c'est tout ce que le LLM reçoit."""
    return f"{rec['id']} [{rec['t']}] {rec['txt']}"


@mcp.tool(name="identifier_interlocuteur")
def identifier_interlocuteur(nom: str, complement: str = "", surnom: str = "", ctx: Context = None) -> str:
    """Désigne la personne qui parle: nom = son prénom, complement = nom de famille (distingue les homonymes), surnom = comment elle veut être appelée. La crée si inconnue."""
    return service.identifier(nom, complement, surnom, trace=trace_mod.from_context(ctx))


@mcp.tool(name="relier")
def relier(personne_a: str, lien: str, personne_b: str, precision: str = "", ctx: Context = None) -> str:
    """personne_a est le <lien> de personne_b. lien: conjoint, parent, enfant, fratrie, demi, beauparent, beauenfant, ami, collegue, voisin, ou autre + precision. « moi » = l'interlocuteur."""
    return service.relier(personne_a, lien, personne_b, precision, trace=trace_mod.from_context(ctx))


@mcp.tool(name="ui_etat")
def ui_etat(activer: str = "?") -> str:
    """Réservé à l'interface d'Aria (non proposé au LLM). Renvoie en JSON les interlocuteurs, l'actif et la consigne du tour. activer: '?' ne change rien, '' = Inconnue, sinon l'identifiant de la personne."""
    return json.dumps(interlocuteurs.etat(activer), ensure_ascii=False)


@mcp.tool(name="memoriser")
def memoriser(fait: str, type: str = "fait", tags: str = "", ctx: Context = None) -> str:
    """Retient une information durable de l'interlocuteur actuel. type: fait, preference, tache, ephemere, ou foyer (concerne toute la maison). tags: mots séparés par des virgules."""
    return service.memoriser(fait, type, tags, trace=trace_mod.from_context(ctx))


@mcp.tool(name="rechercher_souvenir")
def rechercher_souvenir(question: str = "", limite: int = 5) -> str:
    """Cherche ce que l'on sait de l'interlocuteur actuel et du foyer. À appeler avant de répondre sur ses goûts, sa maison, sa famille ou ses affaires."""
    return service.rechercher(question, limite)


@mcp.tool(name="oublier")
def oublier(id: str, ctx: Context = None) -> str:
    """Oublie un souvenir à partir de son identifiant (donné par rechercher_souvenir)."""
    return service.oublier(id, trace=trace_mod.from_context(ctx))


if __name__ == "__main__":
    port = os.environ.get("MEMORY_UI_PORT")
    if port:
        try:
            import ui
            ui.start(store, activity, int(port), os.environ.get("MEMORY_UI_TOKEN") or None)
        except Exception as e:     # la page est optionnelle : une erreur ne doit jamais empêcher le serveur de démarrer
            activity.log("error", where="ui_start", msg=f"{type(e).__name__}: {e}")
            activity.warn(f"page de gestion non démarrée : {e}")
    http_port = os.environ.get("MEMORY_HTTP_PORT")
    http_secret = os.environ.get("MEMORY_HTTP_SECRET")
    if http_port:
        # Canal HTTP pour les accès tiers : même processus, même Store. Sans secret, il n'est PAS ouvert (jamais de canal sans authentification).
        try:
            if not http_secret or len(http_secret) < 16:
                raise ValueError("MEMORY_HTTP_SECRET absent ou trop court")
            import http_facade
            http_facade.start(store, service, int(http_port), http_secret)
            activity.warn(f"canal HTTP des accès tiers sur 127.0.0.1:{http_port}")
        except Exception as e:     # optionnel : une erreur ne doit jamais empêcher le serveur (donc Aria) de démarrer
            activity.log("error", where="http_start", msg=f"{type(e).__name__}: {e}")
            activity.warn(f"canal HTTP non démarré : {e}")
    mcp.run(transport="stdio")
