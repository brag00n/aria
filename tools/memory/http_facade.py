"""Façade HTTP du serveur de mémoire, pour les accès tiers (components/access/). Le MÊME processus et le MÊME Store que la façade stdio d'Aria :
un seul écrivain, un seul verrou, une seule mémoire.

Sécurité (le serveur impose l'identité, jamais le client) :
- écoute sur 127.0.0.1 seulement ;
- chaque requête doit porter `Authorization: Bearer <secret>` (secret propre au démarrage d'Aria, comparé à temps constant) : sans lui, 401 ;
- l'identité vient de l'en-tête `X-Memory-Principal` (clé interne de l'accès) : la personne SYSTÈME correspondante est retrouvée ou créée, le nom suit
  `X-Memory-Name`. Le LLM n'a aucun moyen de la fournir ni de la modifier ;
- seuls les outils memoriser, rechercher_souvenir, oublier et relier sont servis : ni `identifier_interlocuteur` ni `ui_*`.
"""
import hmac
import json
from urllib.parse import quote, unquote

from mcp.server.fastmcp import Context, FastMCP

import audit_trace as trace_mod

PRINCIPAL_HEADER = "x-memory-principal"
PUBLISH_HEADER = "x-memory-publish"   # « 1 » = l'accès peut publier ses propres liens (option de sa clé, posée par Aria : le LLM ne peut pas l'envoyer)
NAME_HEADER = "x-memory-name"          # nom encodé en pourcentage (UTF-8) : un en-tête HTTP ne porte que de l'ASCII


def encode_name(nom):
    """Nom d'un accès (« Caméra ») -> valeur d'en-tête ASCII, pour le client."""
    return quote(str(nom or ""), safe="")
MAX_KEY = 40
MAX_NAME = 60


class Refus(Exception):
    pass


def make_app(store, service, secret):
    """Application Starlette (transport streamable-http de FastMCP) enveloppée d'un contrôle d'authentification."""
    mcp = FastMCP("memoire-tiers", stateless_http=True, json_response=True)

    def principal(ctx):
        """Identifiant de la personne système qui appelle, d'après les en-têtes (déjà authentifiés par le garde). Lève Refus."""
        request = ctx.request_context.request
        if request is None:
            raise Refus("canal HTTP sans requête")
        cle = (request.headers.get(PRINCIPAL_HEADER) or "").strip()[:MAX_KEY]
        nom = unquote(request.headers.get(NAME_HEADER) or "").strip()[:MAX_NAME]
        if not cle or not nom:
            raise Refus("identité de l'accès absente")
        with store.lock:
            person = store.personnes.ensure_system(cle, nom, trace=trace_mod.from_context(ctx))
        if person is None:
            raise Refus("identité de l'accès invalide")
        return person["id"]

    def publish(ctx):
        """L'option « publier ses liens » de la clé de l'accès, d'après l'en-tête posé par Aria (déjà authentifié par le garde)."""
        try:
            return (ctx.request_context.request.headers.get(PUBLISH_HEADER) or "").strip() == "1"
        except Exception:
            return False

    @mcp.tool(name="memoriser")
    def memoriser(fait: str, type: str = "fait", tags: str = "", ctx: Context = None) -> str:
        """Retient une information durable. type: fait, preference, tache, ephemere, ou foyer. tags: mots séparés par des virgules. Ce que tu retiens t'appartient."""
        return _run(ctx, lambda p, tr: service.memoriser(fait, type, tags, principal=p, trace=tr))

    @mcp.tool(name="rechercher_souvenir")
    def rechercher_souvenir(question: str = "", limite: int = 5, ctx: Context = None) -> str:
        """Cherche tes souvenirs et ce que l'on sait des personnes du foyer et de leurs liens. À appeler avant de répondre sur ce que tu as appris."""
        return _run(ctx, lambda p, tr: service.rechercher(question, limite, principal=p))

    @mcp.tool(name="oublier")
    def oublier(id: str, ctx: Context = None) -> str:
        """Oublie un de tes souvenirs à partir de son identifiant (donné par rechercher_souvenir)."""
        return _run(ctx, lambda p, tr: service.oublier(id, principal=p, trace=tr))

    @mcp.tool(name="relier")
    def relier(personne_a: str, lien: str, personne_b: str, precision: str = "", ctx: Context = None) -> str:
        """personne_a est le <lien> de personne_b. lien: conjoint, parent, enfant, fratrie, demi, beauparent, beauenfant, ami, collegue, voisin, ou autre + precision. L'une des deux personnes doit être « moi »."""
        return _run(ctx, lambda p, tr: service.relier(personne_a, lien, personne_b, precision, principal=p, trace=tr, publier=publish(ctx)))

    def _run(ctx, action):
        try:
            pid = principal(ctx)
        except Refus as e:
            return f"Refusé : {e}."
        tr = trace_mod.from_context(ctx)           # échange d'origine, lu dans `_meta` (jamais dans les arguments du LLM)
        with store.lock:
            return action(pid, tr)

    inner = mcp.streamable_http_app()
    expected = f"Bearer {secret}".encode("utf-8")

    async def app(scope, receive, send):
        """Garde d'authentification : tout ce qui n'est pas authentifié est refusé AVANT d'atteindre le moindre outil."""
        if scope["type"] == "http":
            headers = {k.decode("latin-1").lower(): v for k, v in scope.get("headers", [])}
            if not hmac.compare_digest(headers.get("authorization", b""), expected):
                body = json.dumps({"erreur": "authentification requise"}).encode("utf-8")
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await inner(scope, receive, send)

    app.mcp = mcp
    app.inner = inner
    return app


def start(store, service, port, secret):
    """Lance la façade dans un thread démon, sur 127.0.0.1 seulement. Retourne le serveur uvicorn."""
    import asyncio
    import threading

    import uvicorn
    app = make_app(store, service, secret)

    async def run():
        # le gestionnaire de sessions de FastMCP doit tourner pendant toute la vie de l'application (lifespan de l'app interne)
        async with app.inner.router.lifespan_context(app.inner):
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False, lifespan="off"))
            start.server = server
            await server.serve()

    threading.Thread(target=lambda: asyncio.run(run()), name="memoire-http", daemon=True).start()
    return app
