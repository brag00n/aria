"""Magasin de souvenirs : journal JSONL append-only + instantané + index en RAM (bibliothèque standard seulement).

Fichiers (dans <MEMORY_DIR>/<namespace>/) :
  memories.jsonl   instantané, un souvenir par ligne, écrit uniquement par la compaction
  events.jsonl     journal append-only depuis l'instantané (add / touch / update / del)
  archive/AAAA-MM.jsonl   souvenirs oubliés ou périmés (piste d'audit, jamais purgée automatiquement)
  backups/memories-*.jsonl.gz   les 7 dernières sauvegardes de l'instantané

Rejeu au démarrage = instantané puis journal. Toute ligne illisible est ignorée et signalée, jamais fatale.
Module autonome : aucune importation depuis Aria.
"""
import gzip
import hashlib
import heapq
import json
import os
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring
import textproc
from activity import NullActivity

TYPES = ("fait", "preference", "tache", "ephemere")
_TYPE_ALIASES = {
    "fait": "fait", "faits": "fait", "fact": "fait", "info": "fait",
    "preference": "preference", "preferences": "preference", "pref": "preference", "gout": "preference", "gouts": "preference",
    "tache": "tache", "taches": "tache", "task": "tache", "todo": "tache", "rappel": "tache",
    "ephemere": "ephemere", "ephemeres": "ephemere", "temporaire": "ephemere", "temp": "ephemere",
}
MAX_TEXT = 500
MAX_TAGS = 8
COMPACT_EVENTS = 2000           # compaction quand le journal atteint ce nombre d'événements
COMPACT_BYTES = 1024 * 1024     # ... ou cette taille
KEEP_BACKUPS = 7
WRITES_PER_MINUTE = 30          # limite d'écritures (add) par minute pour les appels du LLM : freine un emballement
PURGE_AFTER_HIDDEN_DAYS = 30    # masqué depuis plus longtemps que ça et sous le seuil d'archivage : archivé


def now_ts():
    return time.time()


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def parse_iso(text):
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def normalize_type(value):
    key = textproc.normalize(value or "").replace(" ", "")
    return _TYPE_ALIASES.get(key, "fait")


def parse_tags(value):
    """Tags en chaîne « a, b » ou liste : normalisés, sans doublon, limités."""
    items = value if isinstance(value, (list, tuple)) else str(value or "").replace(";", ",").split(",")
    out = []
    for item in items:
        tag = textproc.normalize(str(item)).replace(" ", "-")
        if tag and tag not in out:
            out.append(tag)
    return out[:MAX_TAGS]


class Store:
    def __init__(self, directory, namespace="default", activity=None, clock=now_ts, lock_instance=True):
        self.ns_dir = Path(directory) / namespace
        self.snapshot = self.ns_dir / "memories.jsonl"
        self.journal = self.ns_dir / "events.jsonl"
        self.archive_dir = self.ns_dir / "archive"
        self.backup_dir = self.ns_dir / "backups"
        self.activity = activity or NullActivity()
        self.clock = clock
        self.lock = threading.RLock()
        self.read_only = False
        self._lockfile = None
        self._seen = {}               # (taille, date) des fichiers tels que nous les avons écrits ou lus
        self._events = 0
        self._recent_writes = []
        self.records = {}
        self.hashes = {}
        self.inv = {}
        self.doclen = {}
        self.by_tag = {}
        self._total_len = 0
        self.last_compact = None

        self.ns_dir.mkdir(parents=True, exist_ok=True)
        for d in (self.archive_dir, self.backup_dir):
            d.mkdir(exist_ok=True)
        if lock_instance:
            self._acquire_instance_lock()
        t0 = time.perf_counter()
        self._load()
        ms = (time.perf_counter() - t0) * 1000
        self.activity.log("start", n=len(self.records), ms=round(ms, 1), read_only=self.read_only,
                          events=self._events, ns=namespace)
        if not self.read_only and (self._events >= COMPACT_EVENTS or self._journal_size() >= COMPACT_BYTES):
            self.compact()

    # ---------------------------------------------------------------- verrou d'instance
    def _acquire_instance_lock(self):
        """Un seul serveur écrit à la fois. flock est libéré par le noyau à la mort du processus (pas de verrou périmé).
        Verrou indisponible (système de fichiers qui ne le gère pas) : on continue sans, en le signalant."""
        path = self.ns_dir / ".lock"
        try:
            f = open(path, "a+")
            try:
                import fcntl
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    self.read_only = True
                    self.activity.log("lock", mode="read_only")
                    self.activity.warn("mémoire verrouillée par un autre processus : lecture seule")
            except ImportError:
                try:
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    self.read_only = True
                    self.activity.log("lock", mode="read_only")
                    self.activity.warn("mémoire verrouillée par un autre processus : lecture seule")
                except ImportError:
                    pass
            self._lockfile = f
        except OSError as e:
            self.activity.warn(f"verrou d'instance indisponible ({e}) : poursuite sans verrou")

    # ---------------------------------------------------------------- chargement
    def _stat(self, path):
        try:
            st = path.stat()
            return (st.st_size, int(st.st_mtime))
        except OSError:
            return None

    def _journal_size(self):
        s = self._stat(self.journal)
        return s[0] if s else 0

    def _remember_files(self):
        self._seen = {"snap": self._stat(self.snapshot), "jrnl": self._stat(self.journal)}

    def _reset_index(self):
        self.records, self.hashes, self.inv, self.doclen, self.by_tag = {}, {}, {}, {}, {}
        self._total_len = 0

    def _load(self):
        self._reset_index()
        self._events = 0
        for line in self._read_lines(self.snapshot):
            rec = self._decode_record(line)
            if rec:
                self._index_add(rec)
        truncated_at = None
        for line, end in self._read_lines_with_offsets(self.journal):
            ev = self._decode_json(line)
            if ev is None:
                continue
            self._events += 1
            self._apply(ev)
        # Dernière ligne sans saut de ligne (écriture interrompue) : on recoupe pour que la suivante soit propre
        if not self.read_only:
            truncated_at = self._truncated_tail_offset()
            if truncated_at is not None:
                try:
                    with open(self.journal, "r+b") as f:
                        f.truncate(truncated_at)
                    self.activity.warn("dernière ligne du journal interrompue : retirée")
                except OSError:
                    pass
        self._remember_files()

    def _truncated_tail_offset(self):
        try:
            with open(self.journal, "rb") as f:
                data = f.read()
        except OSError:
            return None
        if not data or data.endswith(b"\n"):
            return None
        return data.rfind(b"\n") + 1

    def _read_lines(self, path):
        try:
            with open(path, "rb") as f:
                for raw in f:
                    if raw.endswith(b"\n"):
                        yield raw
        except OSError:
            return

    def _read_lines_with_offsets(self, path):
        pos = 0
        for raw in self._read_lines(path):
            pos += len(raw)
            yield raw, pos

    def _decode_json(self, raw):
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except (ValueError, UnicodeDecodeError):
            self.activity.warn("ligne illisible ignorée")
            return None

    def _decode_record(self, raw):
        obj = self._decode_json(raw)
        return self._record_from_json(obj) if obj else None

    def _record_from_json(self, obj):
        try:
            txt = str(obj["txt"]).strip()[:MAX_TEXT]
            if not txt:
                return None
            mtype = normalize_type(obj.get("t", "fait"))
            created = parse_iso(obj.get("c")) or self.clock()
            return {
                "id": str(obj["id"]), "t": mtype, "txt": txt,
                "imp": float(obj.get("imp", scoring.START_IMP[mtype])),
                "tags": parse_tags(obj.get("tags", [])),
                "h": obj.get("h") or self._hash(txt),
                "c": created, "last": parse_iso(obj.get("last")) or created,
                "hits": int(obj.get("hits", 0)), "s": float(obj.get("s", 1.0)),
                "src": str(obj.get("src", "llm")),
            }
        except (KeyError, TypeError, ValueError):
            self.activity.warn("enregistrement invalide ignoré")
            return None

    def _record_to_json(self, rec):
        return {"id": rec["id"], "t": rec["t"], "txt": rec["txt"], "imp": rec["imp"], "tags": rec["tags"],
                "h": rec["h"], "c": iso(rec["c"]), "last": iso(rec["last"]), "hits": rec["hits"],
                "s": round(rec["s"], 3), "src": rec["src"]}

    @staticmethod
    def _hash(text):
        return hashlib.sha256(textproc.normalize(text).encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------------- application des événements
    def _apply(self, ev):
        op = ev.get("op")
        if op == "add":
            rec = self._record_from_json(ev.get("r") or {})
            if rec and rec["id"] not in self.records and rec["h"] not in self.hashes:
                self._index_add(rec)
        elif op == "touch":
            rec = self.records.get(ev.get("id"))
            if rec:
                ts = parse_iso(ev.get("ts")) or self.clock()
                rec["hits"] += 1
                rec["last"] = max(rec["last"], ts)
                rec["s"] = min(scoring.S_MAX, rec["s"] + float(ev.get("boost", scoring.RECALL_BOOST)))
        elif op == "update":
            rec = self.records.get(ev.get("id"))
            if rec:
                self._index_update(rec, ev.get("set") or {})
        elif op == "del":
            rec = self.records.get(ev.get("id"))
            if rec:
                self._index_remove(rec)
        # opération inconnue : ignorée

    # ---------------------------------------------------------------- index
    def _tokens(self, rec):
        """Racines indexées d'un souvenir (texte + tags). Mémorisées : indexer puis désindexer ne normalise qu'une fois."""
        cached = rec.get("_tok")
        if cached is None:
            cached = textproc.analyze(rec["txt"] + " " + " ".join(rec["tags"]))
            rec["_tok"] = cached
        return cached

    def _index_add(self, rec):
        self.records[rec["id"]] = rec
        self.hashes[rec["h"]] = rec["id"]
        tokens = self._tokens(rec)
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            self.inv.setdefault(tok, {})[rec["id"]] = tf
        self.doclen[rec["id"]] = len(tokens)
        self._total_len += len(tokens)
        for tag in rec["tags"]:
            self.by_tag.setdefault(tag, set()).add(rec["id"])

    def _index_remove(self, rec):
        rid = rec["id"]
        self.records.pop(rid, None)
        if self.hashes.get(rec["h"]) == rid:
            self.hashes.pop(rec["h"], None)
        tokens = set(self._tokens(rec))
        for tok in tokens:
            postings = self.inv.get(tok)
            if postings is not None:
                postings.pop(rid, None)
                if not postings:
                    self.inv.pop(tok, None)
        self._total_len -= self.doclen.pop(rid, 0)
        for tag in rec["tags"]:
            ids = self.by_tag.get(tag)
            if ids is not None:
                ids.discard(rid)
                if not ids:
                    self.by_tag.pop(tag, None)

    def _index_update(self, rec, changes):
        """Modification : on retire puis on ré-indexe, ce qui garde l'index toujours cohérent."""
        self._index_remove(rec)
        rec.pop("_tok", None)
        if "txt" in changes:
            txt = str(changes["txt"]).strip()[:MAX_TEXT]
            if txt:
                rec["txt"] = txt
                rec["h"] = self._hash(txt)
        if "t" in changes:
            rec["t"] = normalize_type(changes["t"])
        if "tags" in changes:
            rec["tags"] = parse_tags(changes["tags"])
        if "imp" in changes:
            try:
                rec["imp"] = max(0.0, min(10.0, float(changes["imp"])))
            except (TypeError, ValueError):
                pass
        self._index_add(rec)

    # ---------------------------------------------------------------- écriture du journal
    def _append(self, ev, sync):
        """Un seul write() par ligne. fsync pour les opérations qui comptent (pas pour touch)."""
        line = (json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with open(self.journal, "ab") as f:
            f.write(line)
            f.flush()
            if sync:
                os.fsync(f.fileno())
        self._events += 1
        self._remember_files()

    def _check_external_changes(self):
        """Édition manuelle d'un fichier : si taille ou date diffèrent de ce que nous avons écrit, on recharge."""
        cur = {"snap": self._stat(self.snapshot), "jrnl": self._stat(self.journal)}
        if cur != self._seen:
            self.activity.log("reload", reason="external_change")
            self._load()

    def _writable(self):
        if self.read_only:
            return "mémoire verrouillée par un autre processus."
        return None

    def _rate_limited(self, src):
        """Limite les ajouts émis par le LLM (une page web piégée pourrait lui faire tout mémoriser)."""
        if src != "llm":
            return False
        now = self.clock()
        self._recent_writes = [t for t in self._recent_writes if now - t < 60]
        if len(self._recent_writes) >= WRITES_PER_MINUTE:
            self.activity.log("rate_limit", src=src, limit=WRITES_PER_MINUTE)
            return True
        self._recent_writes.append(now)
        return False

    # ---------------------------------------------------------------- opérations
    def add(self, text, mtype="fait", tags="", src="llm"):
        """Retourne (statut, enregistrement). statut : 'added' | 'duplicate' | 'locked' | 'rate_limit' | 'empty'."""
        with self.lock:
            self._check_external_changes()
            if (err := self._writable()):
                return "locked", None
            text = (text or "").strip()[:MAX_TEXT]
            if not textproc.normalize(text):
                return "empty", None
            h = self._hash(text)
            existing = self.records.get(self.hashes.get(h))
            now = self.clock()
            if existing:
                existing["s"] = min(scoring.S_MAX, existing["s"] + scoring.DUP_BOOST)
                existing["last"] = now
                self._append({"op": "touch", "ts": iso(now), "id": existing["id"], "boost": scoring.DUP_BOOST}, sync=False)
                self.activity.log("add", id=existing["id"], dup=True, src=src, n=len(text))
                return "duplicate", existing
            if self._rate_limited(src):
                return "rate_limit", None
            mtype = normalize_type(mtype)
            rec = {"id": "m_" + secrets.token_hex(3), "t": mtype, "txt": text, "imp": float(scoring.START_IMP[mtype]),
                   "tags": parse_tags(tags), "h": h, "c": now, "last": now, "hits": 0, "s": 1.0, "src": src}
            while rec["id"] in self.records:
                rec["id"] = "m_" + secrets.token_hex(3)
            self._index_add(rec)
            self._append({"op": "add", "ts": iso(now), "r": self._record_to_json(rec)}, sync=True)
            self.activity.log("add", id=rec["id"], t=mtype, imp=rec["imp"], dup=False, src=src, n=len(text))
            self._maybe_compact()
            return "added", rec

    def update(self, rid, changes, src="ui"):
        with self.lock:
            self._check_external_changes()
            if self._writable():
                return False
            rec = self.records.get(rid)
            if not rec:
                return False
            clean = {k: v for k, v in changes.items() if k in ("txt", "t", "tags", "imp")}
            self._index_update(rec, clean)
            self._append({"op": "update", "ts": iso(self.clock()), "id": rid, "set": clean}, sync=True)
            self.activity.log("update", id=rid, fields=sorted(clean), src=src)
            return True

    def delete(self, rid, why="user", src="llm"):
        """Oubli : le souvenir quitte l'index mais reste dans l'archive (jamais perdu en silence)."""
        with self.lock:
            self._check_external_changes()
            if self._writable():
                return False
            rec = self.records.get(rid)
            if not rec:
                return False
            self._archive(rec, why)
            self._index_remove(rec)
            self._append({"op": "del", "ts": iso(self.clock()), "id": rid, "why": why}, sync=True)
            self.activity.log("delete", id=rid, why=why, src=src)
            return True

    def restore(self, rid, src="ui"):
        """Ré-ajoute un souvenir archivé (la dernière version archivée de cet identifiant)."""
        with self.lock:
            self._check_external_changes()
            if self._writable() or rid in self.records:
                return False
            found = None
            for path in sorted(self.archive_dir.glob("*.jsonl")):
                for raw in self._read_lines(path):
                    obj = self._decode_json(raw)
                    if obj and (obj.get("r") or {}).get("id") == rid:
                        found = obj["r"]
            rec = self._record_from_json(found) if found else None
            if not rec or rec["h"] in self.hashes:
                return False
            rec["last"] = self.clock()
            self._index_add(rec)
            self._append({"op": "add", "ts": iso(self.clock()), "r": self._record_to_json(rec)}, sync=True)
            self.activity.log("restore", id=rid, src=src)
            return True

    def _archive(self, rec, why):
        path = self.archive_dir / (datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m") + ".jsonl")
        line = json.dumps({"why": why, "ts": iso(self.clock()), "r": self._record_to_json(rec)},
                          ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    # ---------------------------------------------------------------- lecture / recherche
    def eff(self, rec, now=None):
        return scoring.effective(rec["imp"], rec["s"], rec["last"], now or self.clock(), rec["t"], rec["tags"])

    def state(self, rec, now=None):
        if scoring.is_protected(rec["tags"]):
            return "protege"
        return "actif" if self.eff(rec, now) >= scoring.HIDE_BELOW else "masque"

    def search(self, question, limit=5, touch=True, src="llm"):
        """Souvenirs classés par pertinence. Question vide : les plus importants et récents. Les masqués sont exclus."""
        with self.lock:
            t0 = time.perf_counter()
            self._check_external_changes()
            now = self.clock()
            limit = max(1, min(int(limit or 5), 10))
            tokens = textproc.analyze(question)
            visible = lambda r: self.eff(r, now) >= scoring.HIDE_BELOW
            scored = []
            if tokens:
                n_docs = max(len(self.records), 1)
                avg = self._total_len / n_docs if n_docs else 1.0
                raw = {}
                doclen, k1, b = self.doclen, scoring.K1, scoring.B
                for tok in set(tokens):
                    postings = self.inv.get(tok)
                    if not postings:
                        continue
                    weight = scoring.idf(n_docs, len(postings)) * (k1 + 1)
                    for rid, tf in postings.items():
                        raw[rid] = raw.get(rid, 0.0) + weight * tf / (tf + k1 * (1 - b + b * doclen.get(rid, 1) / avg))
                top = max(raw.values()) if raw else 0.0
                recent_limit = scoring.RECENT_DAYS * 86400
                # L'importance effective ne fait varier le score que de 0,6 à 1,0 : seuls les meilleurs candidats BM25 méritent
                # d'être évalués (marge x8 pour laisser une place aux souvenirs importants un peu moins pertinents).
                pool = heapq.nlargest(max(limit * 8, 40), raw.items(), key=lambda kv: kv[1])
                for rid, value in pool:
                    rec = self.records[rid]
                    eff = self.eff(rec, now)
                    if eff < scoring.HIDE_BELOW:
                        continue
                    scored.append((scoring.final_score(value / top if top else 0.0, eff, (now - rec["last"]) < recent_limit), rec))
            if not scored:
                # Aucune correspondance (ou question vide) : on propose les souvenirs les plus importants
                pool = [r for r in self.records.values() if visible(r)]
                pool.sort(key=lambda r: (self.eff(r, now), r["last"]), reverse=True)
                scored = [(0.0, r) for r in pool]
                fallback = bool(tokens)
            else:
                fallback = False
            top_n = heapq.nlargest(limit, scored, key=lambda x: x[0])      # inutile de trier tous les candidats
            scored = top_n
            results = [r for _, r in top_n]
            if touch and not self.read_only and tokens and not fallback:
                for rec in results:
                    rec["hits"] += 1
                    rec["last"] = now
                    rec["s"] = min(scoring.S_MAX, rec["s"] + scoring.RECALL_BOOST)
                    self._append({"op": "touch", "ts": iso(now), "id": rec["id"]}, sync=False)
            ms = (time.perf_counter() - t0) * 1000
            self.activity.log("search", src=src, q=self.activity.q(question or ""), tok=len(tokens), n_res=len(results),
                              ids=[r["id"] for r in results], fallback=fallback, ms=round(ms, 2))
            self.activity.debug("search_scores", scores=[round(s, 3) for s, _ in scored[:limit]])
            return results, fallback

    def all_records(self, now=None):
        with self.lock:
            self._check_external_changes()
            return sorted(self.records.values(), key=lambda r: r["c"], reverse=True)

    def archived(self, limit=200):
        out = []
        for path in sorted(self.archive_dir.glob("*.jsonl"), reverse=True):
            for raw in self._read_lines(path):
                obj = self._decode_json(raw)
                if obj and obj.get("r"):
                    out.append({"why": obj.get("why", ""), "ts": obj.get("ts", ""), **obj["r"]})
            if len(out) >= limit:
                break
        return out[:limit]

    def stats(self):
        with self.lock:
            now = self.clock()
            by_type = {}
            hidden = protected = 0
            for rec in self.records.values():
                by_type[rec["t"]] = by_type.get(rec["t"], 0) + 1
                st = self.state(rec, now)
                hidden += st == "masque"
                protected += st == "protege"
            return {"total": len(self.records), "par_type": by_type, "masques": hidden, "proteges": protected,
                    "journal_evenements": self._events, "journal_octets": self._journal_size(),
                    "instantane_octets": (self._stat(self.snapshot) or (0,))[0], "derniere_compaction": self.last_compact,
                    "lecture_seule": self.read_only}

    # ---------------------------------------------------------------- compaction et sauvegardes
    def _maybe_compact(self):
        if self._events >= COMPACT_EVENTS or self._journal_size() >= COMPACT_BYTES:
            self.compact()

    def compact(self):
        """Archive les souvenirs trop oubliés, sauvegarde l'ancien instantané, écrit le nouveau (tmp + fsync + replace)
        puis vide le journal. Un crash entre les deux dernières étapes reste correct : les opérations sont idempotentes."""
        with self.lock:
            if self.read_only:
                return None
            t0 = time.perf_counter()
            now = self.clock()
            before = len(self.records)
            archived = 0
            for rec in list(self.records.values()):
                if scoring.is_protected(rec["tags"]):
                    continue
                age_days = (now - rec["last"]) / 86400.0
                if self.eff(rec, now) < scoring.ARCHIVE_BELOW and age_days >= PURGE_AFTER_HIDDEN_DAYS:
                    self._archive(rec, "decay")
                    self._index_remove(rec)
                    archived += 1
            backup = self._backup_snapshot()
            tmp = self.snapshot.with_suffix(".jsonl.tmp")
            with open(tmp, "wb") as f:
                for rec in sorted(self.records.values(), key=lambda r: r["c"]):
                    f.write((json.dumps(self._record_to_json(rec), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.snapshot)
            with open(self.journal, "wb") as f:
                f.flush()
                os.fsync(f.fileno())
            self._events = 0
            self._remember_files()
            self.last_compact = iso(now)
            self.activity.log("compact", before=before, after=len(self.records), archived=archived, backup=backup,
                              ms=round((time.perf_counter() - t0) * 1000, 1))
            return {"before": before, "after": len(self.records), "archived": archived}

    def _backup_snapshot(self):
        if not self.snapshot.exists() or self.snapshot.stat().st_size == 0:
            return None
        stamp = datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = self.backup_dir / f"memories-{stamp}.jsonl.gz"
        n = 1
        while target.exists():
            target = self.backup_dir / f"memories-{stamp}-{n}.jsonl.gz"
            n += 1
        with open(self.snapshot, "rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        backups = sorted(self.backup_dir.glob("memories-*.jsonl.gz"))
        for old in backups[:-KEEP_BACKUPS]:
            try:
                old.unlink()
            except OSError:
                pass
        return target.name

    def close(self):
        if self._lockfile:
            try:
                self._lockfile.close()
            except OSError:
                pass
            self._lockfile = None
