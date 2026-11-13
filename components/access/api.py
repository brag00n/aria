"""API HTTP des applications tierces (Starlette) : POST /v1/turn (flux SSE), /v1/stop, /v1/reset, GET /v1/health.

`create_app(deps)` ne dépend d'aucun module lourd d'Aria : les dépendances sont injectées dans `deps` (`Deps`) ; les tests y mettent un
moteur LLM, un STT et un TTS factices.

Authentification : `Authorization: Bearer ak_…` ou `X-Api-Key`. Une requête de navigateur porte un en-tête `Origin` qui doit figurer
dans la liste de la clé ; la réponse porte alors les en-têtes CORS de cette origine exacte (jamais `*`).
Événements SSE d'un tour : `state` (thinking|speaking|idle), `user_text` (transcription d'un tour vocal), `text` (texte cumulé de la
réponse), `tool` (outil utilisé), `audio` (bloc wav en base64), `done`, `error` (code + message).
"""
import asyncio
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field

import components.utils as utils
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import security
from .profile import AppPromptError, clean_app_prompt

# Fichiers publics : nom d'URL -> (chemin relatif à static/, type MIME). Rien d'autre n'est jamais servi.
PUBLIC_FILES = {
    "aria-client.js": (("aria-client.js",), "application/javascript; charset=utf-8"),
    "silero_vad.onnx": (("silero_v6.2", "silero_vad.onnx"), "application/octet-stream"),
}

TOOL_NOTICE_RE = re.compile(r"\n\n> 🛠️ Outil utilisé : `([^`]+)`\n\n")


@dataclass
class Deps:
    keys: object                     # KeyStore
    sessions: object                 # SessionRegistry
    stt: object = None               # .transcribe_translate(samples float32 16 kHz) -> str ; None = voix indisponible
    tts: object = None               # .run_tts_to_file(text, user_id=…) -> chemin wav ; None = pas de synthèse
    decode_audio: object = None      # (bytes) -> ndarray float32 mono 16 kHz
    stt_lock: object = None          # threading.Lock partagé avec l'interface (STT non réentrant)
    on_audit: object = None          # (événement: str, **champs) -> None
    rate_per_ip: int = 120           # requêtes par minute et par IP
    max_audio_s: float = 30.0
    static_dir: str = ""             # dossier `static/` d'Aria : aria-client.js et le modèle du VAD y sont servis (liste fermée)
    extra: dict = field(default_factory=dict)


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(deps):
    ip_window = security.SlidingWindow(deps.rate_per_ip, 60.0)
    key_window = security.SlidingWindow(30, 60.0)
    fail_lock = security.FailureLock()
    quota = security.DailyQuota()

    def audit(event, **fields):
        if deps.on_audit:
            try:
                deps.on_audit(event, **fields)
            except Exception:
                pass

    def ip_of(request):
        return request.client.host if request.client else "?"

    def error(status, code, message, headers=None):
        return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers or {})

    async def authenticate(request):
        """Retourne (clé, en-têtes CORS, None) ou (None, en-têtes, Response d'erreur)."""
        ip = ip_of(request)
        origin = request.headers.get("origin")
        wait = fail_lock.blocked(ip)
        if wait:
            audit("blocked", ip=ip)
            return None, {}, error(429, "blocked", "trop d'échecs d'authentification", {"Retry-After": str(wait)})
        if not ip_window.allow(ip):
            return None, {}, error(429, "rate_limited", "trop de requêtes", {"Retry-After": str(ip_window.retry_after(ip))})
        status, rec = deps.keys.authenticate(security.bearer(request.headers), origin)
        if status != "ok":
            fail_lock.fail(ip)
            audit("auth_failed", ip=ip, reason=status, origin=origin or "")
            # Le message ne distingue pas « inconnue » de « révoquée » : rien à apprendre pour qui devine des clés
            http = 403 if status == "origin" else 401
            msg = "origine non autorisée pour cette clé" if status == "origin" else "clé API invalide"
            return None, {}, error(http, "forbidden" if status == "origin" else "unauthorized", msg)
        fail_lock.success(ip)
        cors = security.cors_headers(origin, rec["origins"])
        if not key_window.allow(rec["id"], rec["limits"]["rate_per_min"]):
            return None, cors, error(429, "rate_limited", "limite de débit de la clé atteinte",
                                     {**cors, "Retry-After": str(key_window.retry_after(rec["id"]))})
        return rec, cors, None

    async def preflight(request):
        origin = request.headers.get("origin", "")
        # Le contrôle préalable ne porte pas la clé : on autorise l'origine si elle figure dans AU MOINS une clé active
        allowed = set()
        for k in deps.keys.list():
            if k["etat"] == "active":
                allowed.update(k["origins"])
        return Response(status_code=204, headers=security.preflight_headers(origin, allowed))

    async def health(request):
        rec, cors, err = await authenticate(request)
        if err:
            return err
        return JSONResponse({"ok": True, "voice": bool(rec["voice"] and deps.stt), "key": rec["id"]}, headers=cors)

    async def read_json(request, cors):
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > security.MAX_BODY:
            return None, error(413, "too_large", "requête trop volumineuse", cors)
        if "application/json" not in request.headers.get("content-type", ""):
            return None, error(415, "content_type", "Content-Type: application/json attendu", cors)
        raw = await request.body()
        if len(raw) > security.MAX_BODY:
            return None, error(413, "too_large", "requête trop volumineuse", cors)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None, error(400, "bad_json", "JSON invalide", cors)
        if not isinstance(body, dict):
            return None, error(400, "bad_json", "un objet JSON est attendu", cors)
        return body, None

    async def get_session(rec, body, cors, app_prompt=""):
        try:
            return await deps.sessions.get(rec, body.get("session"), app_prompt), None
        except Exception as e:
            code = getattr(e, "code", 500)
            return None, error(code, "session", getattr(e, "message", "session impossible"), cors)

    async def turn(request):
        rec, cors, err = await authenticate(request)
        if err:
            return err
        body, err = await read_json(request, cors)
        if err:
            return err
        text = body.get("text")
        audio = body.get("audio")
        if (text is None) == (audio is None):
            return error(400, "bad_request", "fournissez soit « text », soit « audio »", cors)
        if text is not None:
            text = " ".join(str(text).split())
            if not text:
                return error(400, "bad_request", "texte vide", cors)
            if len(text) > security.MAX_TEXT:
                return error(413, "too_large", f"texte limité à {security.MAX_TEXT} caractères", cors)
        else:
            if not (rec["voice"] and deps.stt and deps.decode_audio):
                return error(403, "voice_disabled", "la voix n'est pas disponible pour cette clé", cors)
            if not isinstance(audio, str) or "," not in audio[:80]:
                return error(400, "bad_request", "audio : data URI base64 attendue", cors)
        try:
            app_prompt = clean_app_prompt(body.get("system"))
        except AppPromptError as e:
            return error(e.status, "bad_request" if e.status == 400 else "too_large", str(e), cors)
        session, err = await get_session(rec, body, cors, app_prompt)
        if err:
            return err
        if not quota.take(rec["id"], rec["limits"]["daily_turns"]):
            audit("quota", key=rec["id"])
            return error(429, "quota", "quota quotidien atteint", cors)
        speak = bool(body.get("speak", True)) and bool(deps.tts) and rec["voice"]
        headers = {**cors, "Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        ignored = bool(app_prompt) and app_prompt != session.app_prompt          # la session existe déjà avec d'autres consignes
        return StreamingResponse(_stream_turn(deps, session, rec, text, audio, speak, audit, ignored), media_type="text/event-stream",
                                 headers=headers)

    async def stop(request):
        rec, cors, err = await authenticate(request)
        if err:
            return err
        body, err = await read_json(request, cors)
        if err:
            return err
        session, err = await get_session(rec, body, cors)
        if err:
            return err
        session.stop.set()
        return JSONResponse({"ok": True}, headers=cors)

    async def reset(request):
        rec, cors, err = await authenticate(request)
        if err:
            return err
        body, err = await read_json(request, cors)
        if err:
            return err
        session, err = await get_session(rec, body, cors)
        if err:
            return err
        deps.sessions.reset(session)
        return JSONResponse({"ok": True}, headers=cors)

    async def asset(request):
        """Fichiers publics du client : le script et ses ressources (liste fermée, jamais de chemin libre). Sans clé : ce sont des fichiers
        publics, utiles avant même d'avoir une clé ; le CORS est ouvert en lecture seule pour que `<script src>` et le chargement du
        modèle fonctionnent depuis n'importe quelle page."""
        name = request.path_params.get("name") or request.url.path.lstrip("/")
        entry = PUBLIC_FILES.get(name)
        if not entry or not deps.static_dir:
            return error(404, "not_found", "introuvable")
        path = os.path.join(deps.static_dir, *entry[0])
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return error(404, "not_found", "introuvable")
        headers = {"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff",
                   "Cross-Origin-Resource-Policy": "cross-origin"}
        return Response(data, media_type=entry[1], headers=headers)

    routes = [Route("/aria-client.js", asset, methods=["GET"]), Route("/assets/{name}", asset, methods=["GET"])]
    for path, handler in (("/v1/health", health), ("/v1/turn", turn), ("/v1/stop", stop), ("/v1/reset", reset)):
        routes.append(Route(path, handler, methods=["GET"] if path.endswith("health") else ["POST"]))
        routes.append(Route(path, preflight, methods=["OPTIONS"]))
    app = Starlette(routes=routes)
    app.state.deps = deps
    return app


async def _stream_turn(deps, session, rec, text, audio, speak, audit, ignored=False):
    """Flux SSE d'un tour. Un seul tour à la fois par session ; « stop » coupe la réponse en cours."""
    if session.lock.locked():
        yield _sse("error", {"code": "busy", "message": "un tour est déjà en cours dans cette session"})
        return
    async with session.lock:
        session.stop.clear()
        session.touch()
        t0 = time.perf_counter()
        try:
            yield _sse("state", {"state": "thinking"})
            if ignored:
                yield _sse("warning", {"code": "system_ignored", "message": "la session existe déjà : ses consignes sont fixées à sa création "
                                                                          "(utilisez une autre session pour changer de consignes)"})
            if audio is not None:
                try:
                    raw = base64.b64decode(audio.split(",", 1)[1], validate=False)
                    samples = await asyncio.get_running_loop().run_in_executor(None, _decode_and_transcribe, deps, raw)
                except _AudioTooLong:
                    yield _sse("error", {"code": "audio_too_long", "message": f"audio limité à {deps.max_audio_s:.0f} s"})
                    return
                except Exception as e:
                    yield _sse("error", {"code": "bad_audio", "message": "audio illisible"})
                    utils.log_info("ACCESS", f"audio illisible ({rec['id']}) : {type(e).__name__}")
                    return
                text = samples
                if not re.search(r"\w", text or ""):
                    yield _sse("done", {"empty": True})
                    return
                yield _sse("user_text", {"text": text})
            audit("turn", key=rec["id"], session=session.sid, voice=audio is not None)
            tts = deps.tts if speak else None
            vu = ""                                   # texte déjà émis, pour n'envoyer que les changements
            outils = set()
            gen = session.engine.get_answer_web(tts, text, session.user)
            try:
                async for full, wav, spoken in gen:
                    if session.stop.is_set():
                        yield _sse("done", {"stopped": True})
                        return
                    propre = TOOL_NOTICE_RE.sub("", full or "")
                    for name in TOOL_NOTICE_RE.findall(full or ""):
                        if name not in outils:
                            outils.add(name)
                            yield _sse("tool", {"name": name})
                    if full and full.startswith("Erreur LLM:"):
                        yield _sse("error", {"code": "llm", "message": "le modèle n'a pas pu répondre"})
                        utils.log_info("ACCESS", f"{rec['id']} : {full[:200]}")
                        return
                    if propre != vu:
                        vu = propre
                        yield _sse("text", {"text": propre})
                    if wav:
                        data = await asyncio.get_running_loop().run_in_executor(None, _read_and_remove, wav)
                        if data:
                            yield _sse("state", {"state": "speaking"})
                            yield _sse("audio", {"id": f"a_{int(time.time_ns())}", "format": "wav", "data": data, "text": spoken or ""})
            finally:
                await gen.aclose()
            yield _sse("done", {"ms": int((time.perf_counter() - t0) * 1000)})
        except asyncio.CancelledError:                # client parti : on arrête proprement
            raise
        except Exception as e:
            utils.log_info("ACCESS", f"❌ tour ({rec['id']}) : {type(e).__name__}: {str(e)[:200]}")
            yield _sse("error", {"code": "internal", "message": "erreur interne"})


class _AudioTooLong(Exception):
    pass


def _decode_and_transcribe(deps, raw):
    """Décodage puis STT, dans un thread : ni l'un ni l'autre ne doit bloquer la boucle asynchrone."""
    samples = deps.decode_audio(raw)
    if len(samples) / 16000.0 > deps.max_audio_s:
        raise _AudioTooLong()
    if deps.stt_lock:
        with deps.stt_lock:
            return deps.stt.transcribe_translate(samples)
    return deps.stt.transcribe_translate(samples)


def _read_and_remove(path):
    """Contenu base64 d'un wav de synthèse, puis suppression du fichier."""
    import os
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return data
    except OSError:
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
