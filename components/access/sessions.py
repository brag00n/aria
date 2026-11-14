"""Sessions Aria des applications tierces : un moteur LLM, un gestionnaire MCP et un historique par (clé, session client).

Chaque session a une TÂCHE PROPRIÉTAIRE qui ouvre les connexions MCP, attend un événement d'arrêt puis les ferme : les bibliothèques
asynchrones des clients MCP (anyio) exigent que les portées d'annulation soient ouvertes et fermées dans la même tâche, ce qui exclut
une connexion paresseuse dans la requête (annulable) et une fermeture depuis une autre tâche.
"""
import asyncio
import time

import components.utils as utils

from .profile import REMOTE_BACKENDS, ProfileError, compose_system


class SessionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class Session:
    def __init__(self, sid, key_id, profile, engine, mcp, app_prompt=""):
        self.sid, self.key_id, self.profile, self.engine, self.mcp = sid, key_id, profile, engine, mcp
        self.app_prompt = app_prompt          # consignes de l'application, fixées à la création de la session
        self.user = sid                       # clé de l'historique dans le moteur
        self.lock = asyncio.Lock()            # un seul tour à la fois dans une session
        self.stop = asyncio.Event()           # « stop » demandé par le client
        self.closed = asyncio.Event()
        self.last_used = time.monotonic()
        self.task = None
        self.ready = asyncio.Event()
        self.error = None

    def touch(self):
        self.last_used = time.monotonic()


class SessionRegistry:
    """`factory(profile) -> (engine, mcp)` construit le moteur d'une session (injectable : les tests passent un moteur factice)."""

    def __init__(self, profiles, factory, max_sessions=8, ttl_s=900, clock=time.monotonic):
        self.profiles, self.factory = profiles, factory
        self.max_sessions, self.ttl_s, self.clock = max_sessions, ttl_s, clock
        self.sessions = {}
        self._lock = None
        self._reaper = None
        self._loop = None

    def _guard(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def sid_of(key_id, client_session):
        return f"{key_id}:{client_session}"

    @staticmethod
    def clean_client(client_session):
        """Identifiant de session fourni par le client : lettres, chiffres, - et _ seulement, 40 caractères au plus."""
        return "".join(c for c in str(client_session or "default") if c.isalnum() or c in "-_")[:40] or "default"

    async def get(self, key_rec, client_session, app_prompt=""):
        """Session de (clé, session client), créée à la demande. `app_prompt` (prompt système de l'application, déjà nettoyé) n'est pris en
        compte QU'À LA CRÉATION : une session existante garde le sien. Lève SessionError (code HTTP dans `.code`)."""
        sid = self.sid_of(key_rec["id"], self.clean_client(client_session))
        self._loop = asyncio.get_running_loop()          # pour request_close_key, appelée depuis un autre thread
        async with self._guard():
            s = self.sessions.get(sid)
            if s and not s.closed.is_set():
                s.touch()
                return s
            self._evict_expired_locked()
            if len(self.sessions) >= self.max_sessions:
                raise SessionError(503, "trop de sessions ouvertes, réessayez plus tard")
            if not key_rec.get("profile"):
                raise SessionError(403, "cette clé n'a pas de profil")
            try:
                profile = self.profiles.load(key_rec["profile"])
            except ProfileError as e:
                raise SessionError(500, str(e))
            profile = dict(profile, app_prompt=app_prompt)
            try:
                engine, mcp = self.factory(profile)
            except Exception as e:
                utils.log_info("ACCESS", f"❌ Session impossible pour la clé {key_rec['id']} : {type(e).__name__}: {str(e)[:200]}")
                raise SessionError(500, "configuration du profil invalide")
            s = Session(sid, key_rec["id"], profile, engine, mcp, app_prompt)
            s.task = asyncio.create_task(self._own(s))
            self.sessions[sid] = s
        try:
            await asyncio.wait_for(s.ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            await self.close(sid)
            raise SessionError(504, "les outils de la session ne répondent pas")
        return s

    async def _own(self, s):
        """Tâche propriétaire : connexion MCP, attente de la fermeture, déconnexion, dans la même tâche."""
        try:
            if s.mcp is not None:
                try:
                    await s.mcp.ensure_connected()
                except Exception as e:
                    s.error = f"{type(e).__name__}: {str(e)[:200]}"
                    utils.log_info("ACCESS", f"⚠️ Connexion MCP de la session {s.sid} : {s.error}")
            s.ready.set()
            await s.closed.wait()
        finally:
            s.ready.set()
            if s.mcp is not None:
                try:
                    await s.mcp.close()
                except Exception as e:
                    utils.log_info("ACCESS", f"Fermeture MCP de {s.sid} : {type(e).__name__}")

    def _evict_expired_locked(self):
        now = self.clock()
        for sid, s in list(self.sessions.items()):
            if s.closed.is_set() or (now - s.last_used > self.ttl_s and not s.lock.locked()):
                self._drop_locked(sid)

    def _drop_locked(self, sid):
        s = self.sessions.pop(sid, None)
        if s:
            s.stop.set()
            s.closed.set()

    async def close(self, sid):
        async with self._guard():
            s = self.sessions.get(sid)
            self._drop_locked(sid)
        if s and s.task:
            try:
                await asyncio.wait_for(s.task, timeout=10)
            except (asyncio.TimeoutError, Exception):
                s.task.cancel()

    async def close_client(self, key_rec, client_session):
        """Ferme la session `client_session` de CETTE clé (jamais celle d'une autre : l'identifiant contient l'identifiant de la clé).
        Ne crée rien. Retourne True si elle existait."""
        sid = self.sid_of(key_rec["id"], self.clean_client(client_session))
        existed = sid in self.sessions
        await self.close(sid)
        return existed

    async def close_key(self, key_id):
        """Ferme toutes les sessions d'une clé (révocation ou changement de profil)."""
        for sid in [sid for sid, s in self.sessions.items() if s.key_id == key_id]:
            await self.close(sid)

    def request_close_key(self, key_id):
        """Ferme les sessions d'une clé depuis un AUTRE thread (page d'administration) : la fermeture est planifiée dans la boucle de l'API."""
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.close_key(key_id), loop)

    async def close_all(self):
        for sid in list(self.sessions):
            await self.close(sid)

    def reset(self, s):
        """Efface l'historique de la session."""
        moteur = getattr(s.engine, "engine", s.engine)            # `Llm` enveloppe le moteur réel (Anthropic, OpenAI...)
        store = getattr(moteur, "user_aware_messages", None)
        if isinstance(store, dict):
            store.pop(s.user, None)

    def start_reaper(self, every=60):
        """Tâche périodique qui ferme les sessions inactives."""
        async def loop():
            while True:
                await asyncio.sleep(every)
                async with self._guard():
                    self._evict_expired_locked()
        self._reaper = asyncio.create_task(loop())

    def stats(self):
        return [{"id": s.sid, "key": s.key_id, "profile": s.profile["name"], "idle_s": int(self.clock() - s.last_used)}
                for s in self.sessions.values()]


def build_engine(profile, base_config, language=None):
    """Moteur LLM et gestionnaire MCP d'une session, à partir du profil et de la config d'Aria (LLM distant uniquement).
    Le module LLM du profil (sinon le module actif d'Aria s'il est distant) fournit modèle, url et clé ; `system_message` du profil
    remplace celui du module ; le MCP vient du profil, jamais de la config d'Aria. Retourne (Llm, Mcp|None)."""
    from components.llm import Llm
    from components.mcp import Mcp
    llm_conf = base_config.get("llm", {})
    name = profile.get("llm") or base_config.get("active_modules", {}).get("llm", "")
    params = dict((llm_conf.get(name) or {}).get("params") or {})
    if params.get("backend_type", "local") not in REMOTE_BACKENDS:
        raise ProfileError("seuls les LLM distants sont autorisés pour un accès tiers")
    # Message système : celui du profil (sinon celui du module LLM), puis les consignes de l'application, qui le complètent
    base = profile.get("system_message") or params.get("system_message", "")
    params["system_message"] = compose_system(base, profile.get("app_prompt", ""))
    memory = base_config.get("memory")
    servers = profile["mcp"]["servers"]
    all_config = {"memory": memory, "mcp": {"params": {"servers": servers}}} if servers else {"memory": memory}
    llm = Llm(params, all_config=all_config, language=language)
    mcp = getattr(llm.engine, "mcp_manager", None)
    return llm, mcp
