"""Clés API des applications tierces d'Aria : création, vérification, révocation. Bibliothèque standard seulement.

Fichier `keys.json` (dossier de données, hors git), écrit de façon atomique :
    {"keys": [{"id":"k_a1b2c3","label":"Portail","prefix":"ak_Xy12","hash":"<hmac>","created":"…","expires":"",
               "revoked":false,"origins":["https://app.example.com"],"profile":"portail","limits":{"rate_per_min":30,"daily_turns":500},
               "voice":true,"last_used":""}]}

La clé en clair (« ak_ » + 43 caractères) n'est montrée qu'à la création et n'est jamais stockée : seul un HMAC-SHA256 l'est, avec un secret
propre à l'installation (`secret.key`, créé au premier usage). Comparaison à temps constant. Les dates sont en secondes Unix (UTC).
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

KEY_PREFIX = "ak_"
DEFAULT_LIMITS = {"rate_per_min": 30, "daily_turns": 500}
MAX_LABEL = 60
MAX_ORIGINS = 20
MAX_PROFILE = 60
MAX_RETENTION_DAYS = 3650


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else ""


def normalize_origin(value):
    """« HTTPS://App.Example.com:443/ » -> « https://app.example.com:443 » ; '' si ce n'est pas une origine http(s) valide."""
    value = str(value or "").strip().rstrip("/").lower()
    if not (value.startswith("http://") or value.startswith("https://")):
        return ""
    reste = value.split("://", 1)[1]
    if not reste or "/" in reste or "?" in reste or "#" in reste or " " in reste or "*" in reste or "@" in reste:
        return ""                                       # ni chemin, ni joker, ni identifiants : une origine exacte seulement
    return value


def _clean_origins(values):
    out = []
    for v in (values or [])[:MAX_ORIGINS]:
        o = normalize_origin(v)
        if o and o not in out:
            out.append(o)
    return out


def _clean_limits(limits):
    out = dict(DEFAULT_LIMITS)
    for k in out:
        try:
            v = int((limits or {}).get(k, out[k]))
            if v > 0:
                out[k] = min(v, 1_000_000)
        except (TypeError, ValueError):
            pass
    return out


def _clean_retention(value):
    """Jours de conservation des échanges de cet accès dans le journal d'audit : 0 = durée par défaut du journal ; sinon 1 à 3650."""
    try:
        v = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(v, MAX_RETENTION_DAYS))


def _clean_text(value, n):
    return " ".join(str(value or "").split())[:n]


class KeyStore:
    def __init__(self, directory, clock=time.time):
        self.dir = Path(directory)
        self.clock = clock
        self.lock = threading.RLock()
        self.path = self.dir / "keys.json"
        self.items = []
        self._last_flush = 0.0
        self._dirty_use = False
        self.dir.mkdir(parents=True, exist_ok=True)
        self._secret = self._load_secret()
        self._load()

    # ------------------------------------------------------------------ fichiers
    def _load_secret(self):
        path = self.dir / "secret.key"
        try:
            data = path.read_bytes()
            if len(data) >= 32:
                return data
        except OSError:
            pass
        data = secrets.token_bytes(32)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        self._chmod(tmp)
        os.replace(tmp, path)
        return data

    @staticmethod
    def _chmod(path):
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass                                        # Windows ou volume qui l'interdit : sans effet

    def _load(self):
        self.items = []
        try:
            obj = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        for k in obj.get("keys", []) if isinstance(obj, dict) else []:
            if isinstance(k, dict) and k.get("id") and k.get("hash"):
                self.items.append({
                    "id": str(k["id"]), "label": _clean_text(k.get("label"), MAX_LABEL), "prefix": str(k.get("prefix", ""))[:12],
                    "hash": str(k["hash"]), "created": float(k.get("created") or 0), "expires": float(k.get("expires") or 0),
                    "revoked": bool(k.get("revoked")), "origins": _clean_origins(k.get("origins")),
                    "profile": _clean_text(k.get("profile"), MAX_PROFILE), "limits": _clean_limits(k.get("limits")),
                    "voice": bool(k.get("voice", True)), "last_used": float(k.get("last_used") or 0),
                    "retention_days": _clean_retention(k.get("retention_days")),
                    "publish_links": bool(k.get("publish_links", False)),
                })

    def _save(self):
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"keys": self.items}, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        self._chmod(tmp)
        os.replace(tmp, self.path)
        self._last_flush = self.clock()
        self._dirty_use = False

    # ------------------------------------------------------------------ outils
    def _digest(self, key):
        return hmac.new(self._secret, key.encode("utf-8"), hashlib.sha256).hexdigest()

    def _find(self, kid):
        return next((k for k in self.items if k["id"] == kid), None)

    @staticmethod
    def view(k, now=None):
        """Vue publique d'une clé (jamais le hash) pour la page d'administration."""
        now = now or time.time()
        etat = "révoquée" if k["revoked"] else "expirée" if (k["expires"] and k["expires"] <= now) else "active"
        return {"id": k["id"], "label": k["label"], "prefix": k["prefix"], "created": _iso(k["created"]), "expires": _iso(k["expires"]),
                "revoked": k["revoked"], "etat": etat, "origins": list(k["origins"]), "profile": k["profile"], "limits": dict(k["limits"]),
                "voice": k["voice"], "last_used": _iso(k["last_used"]), "retention_days": k.get("retention_days", 0),
                "publish_links": bool(k.get("publish_links", False))}

    # ------------------------------------------------------------------ gestion
    def create(self, label, origins=(), profile="", limits=None, expires_days=0, voice=True, retention_days=0, publish_links=False):
        """Crée une clé. Retourne (vue publique, clé EN CLAIR) : la clé en clair n'est plus jamais récupérable ensuite."""
        label = _clean_text(label, MAX_LABEL)
        if not label:
            return None, ""
        with self.lock:
            key = KEY_PREFIX + secrets.token_urlsafe(32)
            now = self.clock()
            kid = "k_" + secrets.token_hex(3)
            while self._find(kid):
                kid = "k_" + secrets.token_hex(3)
            try:
                days = float(expires_days or 0)
            except (TypeError, ValueError):
                days = 0
            rec = {"id": kid, "label": label, "prefix": key[:7], "hash": self._digest(key), "created": now,
                   "expires": now + days * 86400 if days > 0 else 0, "revoked": False, "origins": _clean_origins(origins),
                   "profile": _clean_text(profile, MAX_PROFILE), "limits": _clean_limits(limits), "voice": bool(voice), "last_used": 0,
                   "retention_days": _clean_retention(retention_days), "publish_links": bool(publish_links)}
            self.items.append(rec)
            self._save()
            return self.view(rec, now), key

    def list(self):
        with self.lock:
            now = self.clock()
            return [self.view(k, now) for k in self.items]

    def get(self, kid):
        with self.lock:
            k = self._find(kid)
            return self.view(k, self.clock()) if k else None

    def update(self, kid, **changes):
        """label, origins, profile, limits, voice, retention_days, publish_links, expires_days (0 = sans expiration, calculée depuis maintenant). Retourne la vue ou None."""
        with self.lock:
            k = self._find(kid)
            if not k:
                return None
            if "label" in changes:
                label = _clean_text(changes["label"], MAX_LABEL)
                if not label:
                    return None
                k["label"] = label
            if "origins" in changes:
                k["origins"] = _clean_origins(changes["origins"])
            if "profile" in changes:
                k["profile"] = _clean_text(changes["profile"], MAX_PROFILE)
            if "limits" in changes:
                k["limits"] = _clean_limits(changes["limits"])
            if "voice" in changes:
                k["voice"] = bool(changes["voice"])
            if "retention_days" in changes:
                k["retention_days"] = _clean_retention(changes["retention_days"])
            if "publish_links" in changes:
                k["publish_links"] = bool(changes["publish_links"])
            if "expires_days" in changes:
                try:
                    days = float(changes["expires_days"] or 0)
                except (TypeError, ValueError):
                    days = 0
                k["expires"] = self.clock() + days * 86400 if days > 0 else 0
            self._save()
            return self.view(k, self.clock())

    def regenerate(self, kid):
        """Nouvelle clé API pour le MÊME accès : même identifiant interne (donc même mémoire, mêmes réglages, mêmes domaines), ancienne clé
        invalidée aussitôt. Une clé révoquée ou expirée le reste (la régénération ne la réactive pas). Retourne (vue, nouvelle clé EN CLAIR),
        ou (None, "") si l'accès n'existe pas. La clé en clair n'est montrée qu'ici."""
        with self.lock:
            k = self._find(kid)
            if not k:
                return None, ""
            key = KEY_PREFIX + secrets.token_urlsafe(32)
            k["hash"] = self._digest(key)
            k["prefix"] = key[:7]
            self._save()
            return self.view(k, self.clock()), key

    def revoke(self, kid, revoked=True):
        with self.lock:
            k = self._find(kid)
            if not k:
                return False
            k["revoked"] = bool(revoked)
            self._save()
            return True

    def delete(self, kid):
        with self.lock:
            before = len(self.items)
            self.items = [k for k in self.items if k["id"] != kid]
            if len(self.items) != before:
                self._save()
                return True
            return False

    # ------------------------------------------------------------------ authentification
    def authenticate(self, key, origin=None):
        """Retourne (statut, enregistrement). statut : 'ok' | 'unknown' | 'revoked' | 'expired' | 'origin'.
        `origin` : en-tête Origin de la requête (None ou '' pour un client qui n'est pas un navigateur : accepté). Un navigateur dont
        l'origine ne figure pas dans la liste de la clé est refusé. L'enregistrement n'est retourné que pour 'ok'."""
        key = str(key or "")
        if not key.startswith(KEY_PREFIX) or len(key) > 200:
            return "unknown", None
        digest = self._digest(key)
        with self.lock:
            found = None
            for k in self.items:                        # pas de sortie anticipée : durée indépendante de la position de la clé
                if hmac.compare_digest(k["hash"], digest):
                    found = k
            if not found:
                return "unknown", None
            now = self.clock()
            if found["revoked"]:
                return "revoked", None
            if found["expires"] and found["expires"] <= now:
                return "expired", None
            if origin:
                if normalize_origin(origin) not in found["origins"]:
                    return "origin", None
            found["last_used"] = now
            self._dirty_use = True
            if now - self._last_flush > 60:             # last_used n'est écrit qu'une fois par minute au plus
                self._save()
            return "ok", dict(found)

    def retention_of(self, kid):
        """Jours de conservation exigés pour l'accès `kid` dans le journal d'audit (0 = défaut). Un accès supprimé : 0."""
        with self.lock:
            k = self._find(kid)
            return k.get("retention_days", 0) if k else 0

    def flush(self):
        with self.lock:
            if self._dirty_use:
                self._save()
