"""Protections de l'API tierce : débit par clé et par IP, blocage après échecs d'authentification, quota journalier, CORS.
Bibliothèque standard seulement ; tout est en mémoire (une seule instance d'Aria).
"""
import threading
import time
from collections import deque

from .keys import normalize_origin

MAX_BODY = 8 * 1024 * 1024          # un tour audio de 30 s en WAV 16 kHz mono 32 bits = 1,9 Mo, en base64 ~2,6 Mo
MAX_TEXT = 2000


class SlidingWindow:
    """Au plus `limit` événements par fenêtre de `window` secondes, pour chaque identifiant."""

    def __init__(self, limit, window=60.0, clock=time.monotonic, max_ids=10000):
        self.limit, self.window, self.clock, self.max_ids = limit, window, clock, max_ids
        self.events = {}
        self.lock = threading.Lock()

    def allow(self, ident, limit=None):
        """True et enregistre l'événement s'il reste de la place ; sinon False (sans l'enregistrer)."""
        limit = limit or self.limit
        now = self.clock()
        with self.lock:
            q = self.events.setdefault(ident, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self.events) > self.max_ids:         # borne la mémoire face à une inondation d'identifiants
                self._purge(now)
            return True

    def retry_after(self, ident):
        now = self.clock()
        with self.lock:
            q = self.events.get(ident)
            return max(1, int(self.window - (now - q[0])) + 1) if q else 1

    def _purge(self, now):
        for k in [k for k, q in self.events.items() if not q or now - q[-1] > self.window]:
            del self.events[k]


class FailureLock:
    """Après `max_fail` échecs d'authentification en `window` secondes, l'IP est bloquée `lock_s` secondes."""

    def __init__(self, max_fail=8, window=300.0, lock_s=300.0, clock=time.monotonic):
        self.max_fail, self.window, self.lock_s, self.clock = max_fail, window, lock_s, clock
        self.fails = {}
        self.locked = {}
        self.lock = threading.Lock()

    def blocked(self, ip):
        now = self.clock()
        with self.lock:
            until = self.locked.get(ip, 0)
            if until > now:
                return int(until - now) + 1
            self.locked.pop(ip, None)
            return 0

    def fail(self, ip):
        now = self.clock()
        with self.lock:
            q = self.fails.setdefault(ip, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            q.append(now)
            if len(q) >= self.max_fail:
                self.locked[ip] = now + self.lock_s
                q.clear()
            if len(self.fails) > 10000:
                self.fails.clear()

    def success(self, ip):
        with self.lock:
            self.fails.pop(ip, None)


class DailyQuota:
    """Nombre de tours par clé et par jour UTC."""

    def __init__(self, clock=time.time):
        self.clock = clock
        self.counts = {}
        self.lock = threading.Lock()

    def _day(self):
        return int(self.clock() // 86400)

    def take(self, kid, limit):
        """True et décompte un tour s'il en reste ; False si le quota du jour est atteint."""
        day = self._day()
        with self.lock:
            d, n = self.counts.get(kid, (day, 0))
            if d != day:
                d, n = day, 0
            if n >= limit:
                return False
            self.counts[kid] = (d, n + 1)
            return True

    def used(self, kid):
        day = self._day()
        with self.lock:
            d, n = self.counts.get(kid, (day, 0))
            return n if d == day else 0


def cors_headers(origin, allowed):
    """En-têtes CORS pour une requête venant de `origin` : l'origine EXACTE si elle est autorisée pour la clé, jamais « * ».
    Sans origine autorisée : aucun en-tête CORS (le navigateur bloque la réponse)."""
    o = normalize_origin(origin)
    if o and o in (allowed or ()):
        return {"Access-Control-Allow-Origin": o, "Vary": "Origin", "Access-Control-Allow-Credentials": "false"}
    return {"Vary": "Origin"}


def preflight_headers(origin, allowed):
    h = cors_headers(origin, allowed)
    if "Access-Control-Allow-Origin" in h:
        h.update({"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                  "Access-Control-Allow-Headers": "Authorization, X-Api-Key, Content-Type",
                  "Access-Control-Max-Age": "600"})
    return h


def bearer(headers):
    """Clé API d'une requête : « Authorization: Bearer ak_… » ou « X-Api-Key: ak_… » ('' si absente)."""
    auth = headers.get("Authorization", "") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return (headers.get("X-Api-Key", "") or "").strip()
