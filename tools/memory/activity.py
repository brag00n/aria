"""Journal d'activité du serveur de mémoire : un événement JSON par ligne, fait pour être analysé (jq, Python, analyze.py).

Fichier models/memory/logs/activity.jsonl, rotation à 5 Mo (5 fichiers : activity.jsonl, activity.1.jsonl ... .4.jsonl).
Écriture sans fsync : perdre une ligne de journal est sans conséquence. Une erreur d'écriture ne doit JAMAIS faire échouer
un outil : toutes les exceptions sont avalées.
"""
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LEVELS = {"info": 1, "debug": 0}


class NullActivity:
    """Journal éteint (tests, utilisation sans dossier de logs)."""
    def log(self, ev, **fields): pass
    def debug(self, ev, **fields): pass
    def warn(self, msg): pass
    def q(self, text): return text
    def tail(self, n=200): return []


class Activity:
    def __init__(self, directory, log_text=True, level="info", max_bytes=5 * 1024 * 1024, keep=5):
        self.dir = Path(directory)
        self.path = self.dir / "activity.jsonl"
        self.log_text = log_text
        self.level = LEVELS.get(level, 1)
        self.max_bytes = max_bytes
        self.keep = max(2, keep)
        self._lock = threading.Lock()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.warn(f"journal d'activité indisponible : {e}")

    def q(self, text):
        """Texte à consigner pour une requête : tel quel, ou une empreinte si MEMORY_LOG_TEXT=0."""
        if self.log_text:
            return text
        return "sha:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:10]

    def log(self, ev, **fields):
        self._write(ev, fields)

    def debug(self, ev, **fields):
        if self.level <= 0:
            self._write(ev, fields)

    def warn(self, msg):
        """Avertissement lisible sur stderr (récupéré dans les logs du conteneur ; stdout est réservé au protocole MCP)."""
        try:
            sys.stderr.write(f"[memoire] {msg}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _write(self, ev, fields):
        try:
            record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "ev": ev}
            record.update(fields)
            line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            with self._lock:
                self._rotate_if_needed(len(line))
                with open(self.path, "ab") as f:
                    f.write(line)
        except Exception:
            pass

    def _rotate_if_needed(self, incoming):
        try:
            if self.path.exists() and self.path.stat().st_size + incoming > self.max_bytes:
                oldest = self.dir / f"activity.{self.keep - 1}.jsonl"
                if oldest.exists():
                    oldest.unlink()
                for i in range(self.keep - 2, 0, -1):
                    src = self.dir / f"activity.{i}.jsonl"
                    if src.exists():
                        src.rename(self.dir / f"activity.{i + 1}.jsonl")
                self.path.rename(self.dir / "activity.1.jsonl")
        except OSError:
            pass

    def tail(self, n=200):
        """Dernières lignes du journal courant (pour la page de gestion)."""
        try:
            with open(self.path, "rb") as f:
                lines = f.read().splitlines()[-n:]
        except OSError:
            return []
        out = []
        for raw in lines:
            try:
                out.append(json.loads(raw.decode("utf-8", errors="replace")))
            except ValueError:
                continue
        return out
