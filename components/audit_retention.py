"""Rétention et purge du journal d'audit des échanges (components/echanges.py). Bibliothèque standard seulement.

- Purge AUTOMATIQUE par âge : un fichier par jour, donc un jour trop ancien = un fichier supprimé (aucune réécriture). Un accès tiers peut exiger moins que le
  défaut (`retention_days` de sa clé, retrouvé par l'identifiant interne lu dans `src`) : le fichier est alors réécrit, une fois par passage.
- Purge CIBLÉE (page d'administration) : par date, conversation, accès ou personne (droit à l'effacement). Réécriture atomique du seul fichier concerné.
- Toute purge laisse une ligne dans `purges.jsonl` : quand, par qui (auto/admin), quel critère, combien de lignes et quelle période — JAMAIS de contenu.
Les écritures du journal et les réécritures se partagent le verrou de `echanges` : un échange écrit pendant une purge n'est jamais perdu.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

from components import echanges

PURGES_FILE = "purges.jsonl"
INTERVAL_S = 6 * 3600
MAX_CRITERE = 120


def _parse(ts):
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except (TypeError, ValueError):
        return None


def _day_start(day):
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def valid_day(value):
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def log_purge(par, critere, lignes, fichiers, periode=("", ""), traces=0):
    """Une ligne dans purges.jsonl : sans contenu. Jamais d'exception."""
    try:
        d = echanges.directory()
        if d is None:
            return
        record = {"ts": echanges.iso(), "par": par, "critere": str(critere)[:MAX_CRITERE], "lignes": int(lignes), "fichiers": int(fichiers),
                  "du": periode[0], "au": periode[1], "traces": int(traces)}
        with echanges._lock:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / PURGES_FILE, "ab") as f:
                f.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    except Exception:
        pass


def read_purges(limit=200):
    d = echanges.directory()
    out = []
    if d is None:
        return out
    try:
        with open(d / PURGES_FILE, "rb") as f:
            for raw in f:
                if raw.endswith(b"\n"):
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
    except OSError:
        pass
    return out[-limit:]


def _rewrite(path, keep_lines):
    """Remplace le fichier par les lignes conservées (atomique) ; le supprime s'il n'en reste aucune."""
    if not keep_lines:
        os.remove(path)
        return
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "wb") as f:
        for raw in keep_lines:
            f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _filter_file(path, drop):
    """Retire d'un fichier les lignes pour lesquelles drop(obj) est vrai. Retourne (supprimées, identifiants supprimés, ts min, ts max).
    Les lignes illisibles et la ligne partielle finale sont conservées telles quelles (ce n'est pas à la purge de les juger)."""
    keep, removed, ids = [], 0, []
    lo = hi = ""
    try:
        with open(path, "rb") as f:
            raws = f.readlines()
    except OSError:
        return 0, [], "", ""
    for raw in raws:
        obj = None
        if raw.endswith(b"\n"):
            try:
                obj = json.loads(raw.decode("utf-8"))
            except ValueError:
                obj = None
        if isinstance(obj, dict) and obj.get("id") and drop(obj):
            removed += 1
            ids.append(obj["id"])
            ts = str(obj.get("ts", ""))
            lo, hi = (ts if not lo or ts < lo else lo), (ts if ts > hi else hi)
        else:
            keep.append(raw)
    if removed:
        _rewrite(path, keep)
    return removed, ids, lo, hi


def purge_auto(default_days=None, access_days=None, now=None, par="auto"):
    """Purge par âge. `access_days(acces_id)` -> nombre de jours exigé par cet accès (ou None/0 : le défaut). Retourne
    {lignes, fichiers, ids}. Un jour entier plus vieux que le défaut est supprimé sans lecture ; sinon seules les lignes d'un accès qui exige moins sont retirées."""
    now = time.time() if now is None else now
    default_days = echanges._int(default_days, echanges._settings["retention_days"])
    total = {"lignes": 0, "fichiers": 0, "ids": []}
    cache = {}

    def jours_de(acces):
        if acces not in cache:
            try:
                n = int(access_days(acces)) if access_days and acces else 0
            except (TypeError, ValueError):
                n = 0
            cache[acces] = n if n > 0 else default_days
        return cache[acces]

    exige_moins = None
    with echanges._lock:
        for day, path in echanges.files():
            try:
                fin_du_jour = _day_start(day) + 86400
            except ValueError:
                continue
            if fin_du_jour <= now - default_days * 86400:
                try:
                    n = sum(1 for _ in echanges.read_lines(path))
                    ids = [o["id"] for o in echanges.read_lines(path)]
                    os.remove(path)
                except OSError:
                    continue
                total["lignes"] += n
                total["fichiers"] += 1
                total["ids"].extend(ids)
                continue
            if access_days is None:
                continue

            def trop_vieille(obj):
                acces = echanges.acces_id_of(obj.get("src"))
                if not acces:
                    return False
                debut = _parse(obj.get("ts"))
                return debut is not None and debut <= now - jours_de(acces) * 86400
            n, ids, _, _ = _filter_file(path, trop_vieille)
            if n:
                total["lignes"] += n
                total["ids"].extend(ids)
                total["fichiers"] += 1 if not path.exists() else 0
    if total["lignes"] or total["fichiers"]:
        log_purge(par, f"âge (défaut {default_days} j)", total["lignes"], total["fichiers"])
    return total


def purge_targeted(date_avant=None, conversation=None, acces=None, personne=None, par="admin"):
    """Purge ciblée : toutes les conditions données doivent être vraies (ET). `date_avant` 'AAAA-MM-JJ' : échanges commencés AVANT ce jour (exclu) ;
    `conversation` : identifiant exact ; `acces` : identifiant interne (« k_2b7cfa ») ou « aria » ; `personne` : interlocuteur (`who`), insensible à la casse.
    Au moins un critère est exigé (jamais de purge totale par accident). Retourne {lignes, fichiers, ids, periode}."""
    criteres = []
    if date_avant:
        if not valid_day(date_avant):
            raise ValueError("date invalide (AAAA-MM-JJ)")
        criteres.append(f"avant {date_avant}")
    if conversation:
        criteres.append(f"conversation {str(conversation)[:60]}")
    if acces:
        criteres.append(f"accès {str(acces)[:40]}")
    if personne:
        criteres.append("personne")                           # le nom d'une personne n'est pas recopié dans le journal des purges
    if not criteres:
        raise ValueError("au moins un critère est exigé")
    limite = _day_start(date_avant) if date_avant else None
    nom = " ".join(str(personne or "").split()).casefold()

    def drop(obj):
        if limite is not None:
            debut = _parse(obj.get("ts"))
            if debut is None or debut >= limite:
                return False
        if conversation and obj.get("conv") != conversation:
            return False
        if acces:
            src = echanges.acces_id_of(obj.get("src")) or "aria"
            if src != acces:
                return False
        if nom and " ".join(str(obj.get("who", "")).split()).casefold() != nom:
            return False
        return True

    total = {"lignes": 0, "fichiers": 0, "ids": [], "periode": ("", "")}
    lo = hi = ""
    with echanges._lock:
        for day, path in echanges.files():
            if limite is not None and _day_start(day) >= limite:
                continue
            n, ids, a, b = _filter_file(path, drop)
            if n:
                total["lignes"] += n
                total["ids"].extend(ids)
                total["fichiers"] += 1 if not path.exists() else 0
                lo, hi = (a if not lo or a < lo else lo), (b if b > hi else hi)
    total["periode"] = (lo, hi)
    log_purge(par, " + ".join(criteres), total["lignes"], total["fichiers"], total["periode"])
    return total


def volumes():
    """Volumes par jour pour la page d'administration : [{jour, lignes, octets}] (aucun contenu)."""
    out = []
    for day, path in echanges.files():
        try:
            octets = path.stat().st_size
            with open(path, "rb") as f:
                lignes = sum(1 for raw in f if raw.endswith(b"\n"))
        except OSError:
            continue
        out.append({"jour": day, "lignes": lignes, "octets": octets})
    return out


class Planificateur:
    """Purge automatique au démarrage puis toutes les 6 h (thread démon). `reglages()` -> (jours par défaut, access_days) lus À CHAQUE passage :
    un changement dans l'onglet Paramètres ou sur une clé est pris en compte sans redémarrer."""

    def __init__(self, reglages, apres=None, interval=INTERVAL_S):
        self.reglages, self.apres, self.interval = reglages, apres, interval
        self._stop = threading.Event()
        self.thread = None
        self.dernier = None

    def passe(self):
        try:
            jours, access_days = self.reglages()
            self.dernier = purge_auto(jours, access_days)
            if self.apres and self.dernier["ids"]:
                self.apres(self.dernier["ids"])
        except Exception:
            self.dernier = None
        return self.dernier

    def start(self):
        def boucle():
            self.passe()
            while not self._stop.wait(self.interval):
                self.passe()
        self.thread = threading.Thread(target=boucle, name="audit-purge", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
