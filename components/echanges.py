"""Journal d'audit des échanges : une ligne par échange (question + réponse), un fichier par jour. Bibliothèque standard seulement.

Fichier `models/audit/echanges-AAAA-MM-JJ.jsonl` (hors git). Un fichier PAR JOUR, jamais de rotation par taille : une rotation ferait perdre des échanges, ce qu'un audit
ne tolère pas, et la purge par âge devient la suppression de fichiers entiers (components/audit_retention.py).

Ligne :
    {"id":"e_…","ts":"début ISO UTC","fin":"ISO UTC","conv":"…","src":"aria" | "k_2b7cfa:Robot","who":"…","q":"…","voix":false,"r":"…",
     "st":"ok|interrompu|erreur","outils":["memoriser"],"ms":1234}
`src` : `aria` pour l'interface ; pour un accès tiers, son identifiant interne suivi de son nom (le nom est celui de l'accès au moment de l'échange).

L'échange courant est un `contextvar` (valeur PAR TÂCHE asyncio : des tours concurrents ne se mélangent pas), posé une seule fois par source. Il alimente aussi le `_meta`
des appels d'outils de mémoire (components/mcp.py) : le LLM ne peut ni le fournir ni le modifier. Une erreur d'écriture ne doit JAMAIS faire échouer un tour.
"""
import contextvars
import hashlib
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RETENTION_DAYS = 90
EXTRAIT_MAX = 200                   # taille de l'extrait de la question conservé dans la trace d'un souvenir ou d'une personne
TOOL_NOTICE_RE = re.compile(r"\n*> 🛠️ Outil utilisé : `[^`]*`\n*")
EXCHANGE_ID_RE = re.compile(r"^e_[0-9a-z]{6,40}$")
FILE_RE = re.compile(r"^echanges-(\d{4}-\d{2}-\d{2})\.jsonl$")

_current = contextvars.ContextVar("aria_echange", default=None)
_lock = threading.Lock()
_settings = {"dir": None, "enabled": True, "log_text": True, "retention_days": DEFAULT_RETENTION_DAYS}


def configure(directory, enabled=True, log_text=True, retention_days=DEFAULT_RETENTION_DAYS):
    """Réglages du journal (une fois au démarrage, ou depuis les tests)."""
    _settings.update(dir=Path(directory) if directory else None, enabled=bool(enabled), log_text=bool(log_text),
                     retention_days=_int(retention_days, DEFAULT_RETENTION_DAYS))


def _int(value, default):
    try:
        v = int(value)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def directory():
    return _settings["dir"]


def new_id():
    """Identifiant d'échange : horodatage (tri chronologique) + aléa. Valide pour EXCHANGE_ID_RE."""
    return f"e_{int(time.time() * 1000):x}{secrets.token_hex(3)}"


def valid_id(value):
    return isinstance(value, str) and bool(EXCHANGE_ID_RE.match(value))


def iso(ts=None):
    return datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc).isoformat(timespec="milliseconds")


def source_label(acces_id=None, acces_nom=None):
    """`aria` sans accès tiers ; sinon « identifiant:nom » (ex. k_2b7cfa:Robot)."""
    if not acces_id:
        return "aria"
    nom = " ".join(str(acces_nom or "").split())[:60]
    return f"{acces_id}:{nom}" if nom else str(acces_id)


def acces_id_of(src):
    """Identifiant interne d'un accès tiers lu dans `src` ('' pour Aria)."""
    src = str(src or "")
    return "" if src == "aria" or not src else src.split(":", 1)[0]


class Echange:
    """Un échange en cours : métadonnées connues avant le tour, question, réponse, outils utilisés."""

    def __init__(self, src, who, conv, voix=False, query=""):
        self.id = new_id()
        self.src, self.who, self.conv, self.voix, self.query = src, who or "", conv or "", bool(voix), query or ""
        self.ts = time.time()
        self.outils = []

    def meta(self):
        """Ce que le serveur de mémoire reçoit dans `_meta` : l'identifiant et ce qu'il faut pour reconstruire la trace (jamais fourni par le LLM)."""
        return {"echange": self.id, "ts": iso(self.ts), "src": self.src, "who": self.who, "x": extrait(self.query)}


def extrait(texte, n=EXTRAIT_MAX):
    """Extrait court d'une question (une seule ligne), pour la trace d'un souvenir."""
    t = " ".join(str(texte or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def start(src, who, conv, voix=False, query=""):
    """Ouvre l'échange courant de CETTE tâche asyncio. À appeler une fois par tour et par source, avant l'appel du moteur."""
    e = Echange(src, who, conv, voix, query)
    _current.set(e)
    return e


def current():
    return _current.get()


def clear():
    _current.set(None)


def current_meta():
    """`_meta` à joindre à un appel d'outil de mémoire, ou None hors échange."""
    e = _current.get()
    return e.meta() if e is not None else None


def note_tool(name):
    e = _current.get()
    if e is not None and name and name not in e.outils:
        e.outils.append(name)


def clean_answer(texte):
    """Réponse sans les lignes « outil utilisé » insérées pour l'affichage."""
    return TOOL_NOTICE_RE.sub("", texte or "").strip()


def _text(valeur):
    """Texte consigné : tel quel, ou une empreinte si la capture du texte est désactivée (comme MEMORY_LOG_TEXT=0)."""
    if _settings["log_text"]:
        return valeur
    return "sha:" + hashlib.sha256((valeur or "").encode("utf-8")).hexdigest()[:10] + f":{len(valeur or '')}c"


def write(e, reponse, statut="ok"):
    """Écrit la ligne de l'échange `e` (une seule fois). Jamais d'exception."""
    try:
        if not _settings["enabled"] or _settings["dir"] is None or e is None:
            return False
        now = time.time()
        record = {"id": e.id, "ts": iso(e.ts), "fin": iso(now), "conv": e.conv, "src": e.src, "who": e.who, "q": _text(e.query),
                  "voix": e.voix, "r": _text(clean_answer(reponse)), "st": statut, "outils": list(e.outils), "ms": int((now - e.ts) * 1000)}
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        day = datetime.fromtimestamp(e.ts, timezone.utc).strftime("%Y-%m-%d")
        with _lock:
            _settings["dir"].mkdir(parents=True, exist_ok=True)
            with open(_settings["dir"] / f"echanges-{day}.jsonl", "ab") as f:
                f.write(line)
        return True
    except Exception:
        return False


def files(dir_=None):
    """Fichiers du journal, du plus ancien au plus récent : [(jour 'AAAA-MM-JJ', chemin)]."""
    d = Path(dir_) if dir_ else _settings["dir"]
    if d is None or not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = FILE_RE.match(p.name)
        if m:
            out.append((m.group(1), p))
    return sorted(out)


def read_lines(path):
    """Lignes complètes d'un fichier du journal ; la dernière ligne partielle (sans retour à la ligne) est ignorée."""
    out = []
    try:
        with open(path, "rb") as f:
            for raw in f:
                if not raw.endswith(b"\n"):
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except ValueError:
                    continue
                if isinstance(obj, dict) and obj.get("id"):
                    out.append(obj)
    except OSError:
        pass
    return out


class ConversationAria:
    """Identifiant de la conversation courante de l'INTERFACE d'Aria (l'interface n'en a pas d'autre). Un nouvel identifiant à chaque début de conversation :
    ouverture par mot-clé, clôture, « Nouvelle conversation », changement d'interlocuteur connu (l'historique du LLM est alors effacé), redémarrage."""

    def __init__(self):
        self.n = 0
        self.id = ""
        self.renew()

    def renew(self):
        self.n += 1
        self.id = "aria-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{self.n}"
        return self.id
