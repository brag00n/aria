"""Façade HTTP du serveur de mémoire, pour les accès tiers (components/access/). Le MÊME processus et le MÊME Store que la façade stdio d'Aria :
un seul écrivain, un seul verrou, une seule mémoire.

Sécurité (le serveur impose l'identité, jamais le client) :
- écoute sur 127.0.0.1 seulement ;
- chaque requête doit porter `Authorization: Bearer <secret>` (secret propre au démarrage d'Aria, comparé à temps constant) : sans lui, 401 ;
- l'identité vient de l'en-tête `X-Memory-Principal` (clé interne de l'accès) : la personne SYSTÈME correspondante est retrouvée ou créée, le nom suit
  `X-Memory-Name`. Le LLM n'a aucun moyen de la fournir ni de la modifier ;
- seuls les outils memoriser, rechercher_souvenir, oublier et relier sont servis : ni `identifier_interlocuteur` ni `ui_*`.

Deuxième canal, `/mcp-i`, pour les accès dont la clé a le droit « publier les informations de son interlocuteur » : l'accès agit alors au nom de
la PERSONNE qui lui parle, avec les droits d'Aria (souvenirs à son nom, lien entre n'importe quelles personnes, fiches créées sans validation).
L'interlocuteur est propre à la SESSION (en-tête `X-Memory-Session`) et n'est JAMAIS `store.active` : un tiers ne change pas à qui parle Aria. Il est
fourni par l'application (`X-Memory-Speaker`, posé par Aria à la création de la session : le LLM ne peut pas le changer) ou, à défaut, désigné par le LLM avec
`identifier_interlocuteur`. Les observations propres de l'accès restent dans sa mémoire privée (`noter_observation`). Aucune authentification de la personne
n'est faite : celui qui peut parler à l'application peut se dire quelqu'un d'autre (à réserver aux applications de confiance).
"""
import hmac
import json
from urllib.parse import quote, unquote

from mcp.server.fastmcp import Context, FastMCP

import audit_trace as trace_mod

PRINCIPAL_HEADER = "x-memory-principal"
PUBLISH_HEADER = "x-memory-publish"   # « 1 » = l'accès peut publier ses propres liens (option de sa clé, posée par Aria : le LLM ne peut pas l'envoyer)
NAME_HEADER = "x-memory-name"          # nom encodé en pourcentage (UTF-8) : un en-tête HTTP ne porte que de l'ASCII
SESSION_HEADER = "x-memory-session"    # identifiant de la session de l'accès (canal /mcp-i) : l'interlocuteur est propre à la session
SPEAKER_HEADER = "x-memory-speaker"    # interlocuteur fourni par l'application : JSON {"n": prénom, "c": nom de famille} encodé en pourcentage
SPEAKER_PATH = "/mcp-i"
MAX_SPEAKERS = 1000                    # sessions suivies ; au-delà, les plus anciennes sont oubliées (l'interlocuteur est alors à redésigner)


def encode_name(nom):
    """Nom d'un accès (« Caméra ») -> valeur d'en-tête ASCII, pour le client."""
    return quote(str(nom or ""), safe="")
MAX_KEY = 40
MAX_NAME = 60


class Refus(Exception):
    pass


def encode_speaker(prenom, famille=""):
    """Interlocuteur fourni par l'application -> valeur d'en-tête ASCII (JSON encodé en pourcentage), pour le client."""
    return quote(json.dumps({"n": str(prenom or "")[:60], "c": str(famille or "")[:60]}, ensure_ascii=False, separators=(",", ":")), safe="")


def make_app(store, service, secret):
    """Application Starlette (transport streamable-http de FastMCP) enveloppée d'un contrôle d'authentification."""
    mcp = FastMCP("memoire-tiers", stateless_http=True, json_response=True)
    mcp_i = FastMCP("memoire-interlocuteur", stateless_http=True, json_response=True, streamable_http_path=SPEAKER_PATH)
    speakers = {}          # (personne système, session) -> identifiant de l'interlocuteur de cette session

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

    # ------------------------------------------------------------------ canal /mcp-i : l'accès parle au nom de son interlocuteur
    def header(ctx, name):
        try:
            return ctx.request_context.request.headers.get(name) or ""
        except Exception:
            return ""

    def provided_speaker(ctx):
        """Interlocuteur fourni par l'application (en-tête posé par Aria) : (prénom, nom de famille) ou None."""
        raw = header(ctx, SPEAKER_HEADER)
        if not raw:
            return None
        try:
            obj = json.loads(unquote(raw))
            return (str(obj.get("n", "")).strip()[:MAX_NAME], str(obj.get("c", "")).strip()[:MAX_NAME]) if isinstance(obj, dict) else None
        except ValueError:
            return None

    def speaker_of(ctx, system, tr, create=True):
        """Interlocuteur de CETTE session : celui déjà retenu, sinon celui que l'application a fourni (retrouvé ou créé comme le fait Aria). '' si inconnu."""
        sid = header(ctx, SESSION_HEADER).strip()[:80]
        key = (system, sid)
        with store.lock:
            pid = speakers.get(key, "")
            if pid and store.personnes.get(pid):
                return pid
            speakers.pop(key, None)
            given = provided_speaker(ctx) if create else None
            if not given or not given[0]:
                return ""
            status, person, _ = store.personnes.resolve(given[0], given[1], "", trace=tr)
            if person is None or store.is_system(person["id"]):
                return ""
            remember(key, person["id"])
            return person["id"]

    def remember(key, pid):
        if len(speakers) >= MAX_SPEAKERS and key not in speakers:
            speakers.pop(next(iter(speakers)), None)
        speakers[key] = pid

    def _run_i(ctx, action):
        try:
            system = principal(ctx)
        except Refus as e:
            return f"Refusé : {e}."
        tr = trace_mod.from_context(ctx)
        with store.lock:
            return action(system, speaker_of(ctx, system, tr), tr)

    @mcp_i.tool(name="identifier_interlocuteur")
    def identifier_interlocuteur_i(nom: str, complement: str = "", surnom: str = "", ctx: Context = None) -> str:
        """Désigne la personne qui te parle: nom = son prénom, complement = nom de famille (distingue les homonymes), surnom = comment elle veut être appelée. La crée si inconnue."""
        def action(system, speaker, tr):
            if provided_speaker(ctx):
                return "L'application a déjà identifié ton interlocuteur : tu ne peux pas le changer."
            text, pid = service.i.identifier_session(nom, complement, surnom, courant=speaker, trace=tr)
            if pid and pid != speaker:
                remember((system, header(ctx, SESSION_HEADER).strip()[:80]), pid)
            return text
        return _run_i(ctx, action)

    @mcp_i.tool(name="memoriser")
    def memoriser_i(fait: str, type: str = "fait", tags: str = "", ctx: Context = None) -> str:
        """Retient une information durable de la personne qui te parle. type: fait, preference, tache, ephemere, ou foyer (concerne toute la maison). tags: mots séparés par des virgules."""
        return _run_i(ctx, lambda system, speaker, tr: service.memoriser_pour(speaker, fait, type, tags, trace=tr))

    @mcp_i.tool(name="noter_observation")
    def noter_observation(fait: str, tags: str = "", ctx: Context = None) -> str:
        """Retient une observation qui t'appartient (mesure, état, constat de ta part), pas une information donnée par la personne. Elle reste dans ta mémoire privée."""
        return _run_i(ctx, lambda system, speaker, tr: service.memoriser(fait, "fait", tags, principal=system, trace=tr))

    @mcp_i.tool(name="rechercher_souvenir")
    def rechercher_souvenir_i(question: str = "", limite: int = 5, ctx: Context = None) -> str:
        """Cherche ce que l'on sait de la personne qui te parle, du foyer et de tes propres observations. À appeler avant de répondre sur ses goûts, sa famille ou ses affaires."""
        return _run_i(ctx, lambda system, speaker, tr: service.rechercher_pour(speaker, system, question, limite))

    @mcp_i.tool(name="oublier")
    def oublier_i(id: str, ctx: Context = None) -> str:
        """Oublie un souvenir à partir de son identifiant (donné par rechercher_souvenir)."""
        return _run_i(ctx, lambda system, speaker, tr: service.oublier_pour(speaker, system, id, trace=tr))

    @mcp_i.tool(name="relier")
    def relier_i(personne_a: str, lien: str, personne_b: str, precision: str = "", ctx: Context = None) -> str:
        """personne_a est le <lien> de personne_b. lien: conjoint, parent, enfant, fratrie, demi, beauparent, beauenfant, ami, collegue, voisin, ou autre + precision. « moi » = la personne qui te parle."""
        return _run_i(ctx, lambda system, speaker, tr: service.relier_pour(speaker, personne_a, lien, personne_b, precision, trace=tr))

    def _run(ctx, action):
        try:
            pid = principal(ctx)
        except Refus as e:
            return f"Refusé : {e}."
        tr = trace_mod.from_context(ctx)           # échange d'origine, lu dans `_meta` (jamais dans les arguments du LLM)
        with store.lock:
            return action(pid, tr)

    inner = mcp.streamable_http_app()
    inner_i = mcp_i.streamable_http_app()
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
        await (inner_i if scope["type"] == "http" and scope.get("path", "").rstrip("/") == SPEAKER_PATH else inner)(scope, receive, send)

    app.mcp = mcp
    app.mcp_i = mcp_i
    app.speakers = speakers
    app.inner = inner
    app.inner_i = inner_i
    return app


def start(store, service, port, secret):
    """Lance la façade dans un thread démon, sur 127.0.0.1 seulement. Retourne le serveur uvicorn."""
    import asyncio
    import threading

    import uvicorn
    app = make_app(store, service, secret)

    async def run():
        # le gestionnaire de sessions de FastMCP doit tourner pendant toute la vie de l'application (lifespan de l'app interne)
        async with app.inner.router.lifespan_context(app.inner), app.inner_i.router.lifespan_context(app.inner_i):
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False, lifespan="off"))
            start.server = server
            await server.serve()

    threading.Thread(target=lambda: asyncio.run(run()), name="memoire-http", daemon=True).start()
    return app
