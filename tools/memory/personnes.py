"""Personnes connues d'Aria (les interlocuteurs) : un fichier JSONL, ajout en fin de fichier, réécriture atomique pour modifier.

Fichier personnes.jsonl (dans le dossier de l'espace de noms), une personne par ligne :
    {"id":"p1","nom":"Olivier","compl":"Martin","surnom":"Oli","c":"2026-10-30T14:02:11+00:00"}

`nom` est le prénom (ou ce que la personne dit être son nom), `compl` un complément qui la distingue d'un homonyme (nom de famille),
`surnom` un nom d'usage. Module autonome : aucune importation depuis Aria, bibliothèque standard seulement.
"""
import json
import os
import threading
from pathlib import Path

import textproc

MAX_FIELD = 60


def _clean(value):
    return " ".join(str(value or "").split())[:MAX_FIELD]


def key(value):
    """Forme de comparaison : minuscules, sans accents ni ponctuation (« Éric » = « eric »)."""
    return textproc.normalize(value)


class Personnes:
    def __init__(self, directory, iso, clock, activity=None):
        self.path = Path(directory) / "personnes.jsonl"
        self.iso = iso
        self.clock = clock
        self.activity = activity
        self.lock = threading.RLock()
        self.items = []
        self._load()

    # ------------------------------------------------------------------ fichier
    def _load(self):
        self.items = []
        try:
            with open(self.path, "rb") as f:
                for raw in f:
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get("id") and obj.get("nom"):
                        self.items.append({"id": str(obj["id"]), "nom": _clean(obj["nom"]), "compl": _clean(obj.get("compl")),
                                           "surnom": _clean(obj.get("surnom")), "c": obj.get("c", "")})
        except OSError:
            pass

    def _append(self, person):
        line = json.dumps(person, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(self.path, "ab") as f:
            f.write(line.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())

    def _rewrite(self):
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "wb") as f:
            for p in self.items:
                f.write((json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ lecture
    def get(self, pid):
        return next((p for p in self.items if p["id"] == pid), None)

    def label(self, person):
        """Libellé court : « Olivier » ; « Olivier (Martin) » dès qu'un homonyme existe, pour les distinguer dans la liste."""
        same_first = [p for p in self.items if key(p["nom"]) == key(person["nom"])]
        extra = person["compl"] or person["surnom"]
        return f"{person['nom']} ({extra})" if (len(same_first) > 1 and extra) else person["nom"]

    def full(self, person):
        parts = [person["nom"]]
        if person["compl"]:
            parts.append(person["compl"])
        if person["surnom"]:
            parts.append(f"dit {person['surnom']}")
        return " ".join(parts)

    def listing(self):
        with self.lock:
            return sorted(({"id": p["id"], "label": self.label(p)} for p in self.items), key=lambda x: key(x["label"]))

    def find(self, name):
        """Personnes dont le nom, le nom complet ou le surnom correspond (comparaison sans casse ni accents)."""
        wanted = key(name)
        if not wanted:
            return []
        out = []
        for p in self.items:
            candidates = {key(p["nom"]), key(p["surnom"]), key(f"{p['nom']} {p['compl']}"), key(f"{p['compl']} {p['nom']}"),
                          key(f"{p['nom']} {p['surnom']}")}
            candidates.discard("")
            if wanted in candidates:
                out.append(p)
        return out

    # ------------------------------------------------------------------ résolution d'un nom donné par le LLM
    def resolve(self, nom, complement="", surnom=""):
        """Retourne (statut, personne, candidats). statut :
          'found'      une seule personne correspond : à activer
          'created'    personne inconnue : créée
          'ambiguous'  plusieurs personnes (ou un prénom déjà pris sans complément) : on n'active rien
          'empty'      nom vide
        Un complément qui ne correspond à aucune personne portant ce prénom crée un homonyme."""
        status, person, candidates = self._resolve(_clean(nom), _clean(complement))
        # Le surnom d'usage (« appelle-moi Oli ») est retenu sur la personne désignée, sans écraser un surnom existant
        surnom = _clean(surnom)
        if person and surnom and not person["surnom"] and key(surnom) != key(person["nom"]):
            person["surnom"] = surnom
            with self.lock:
                self._rewrite()
            if self.activity:
                self.activity.log("person_nickname", id=person["id"])
        return status, person, candidates

    def _resolve(self, nom, complement):
        if not key(nom):
            return "empty", None, []
        with self.lock:
            same_first = [p for p in self.items if key(p["nom"]) == key(nom)]
            # Le nom donné peut déjà être complet (« Olivier Martin ») ou un surnom : on essaie d'abord tel quel
            exact = self.find(nom if not complement else f"{nom} {complement}")
            if complement and exact:
                return "found", exact[0], []
            if not complement:
                direct = self.find(nom)
                if len(direct) == 1 and len(same_first) <= 1:
                    return "found", direct[0], []
                if len(direct) == 1 and direct[0]["surnom"] and key(direct[0]["surnom"]) == key(nom):
                    return "found", direct[0], []         # le surnom désigne une seule personne sans ambiguïté
                if len(same_first) >= 1:
                    return "ambiguous", None, same_first
                # nom complet donné d'un coup : « Olivier Martin » -> prénom + complément
                parts = nom.split(" ", 1)
                if len(parts) == 2:
                    return self._create_or_match(parts[0], parts[1])
                return self._create(nom, "")
            return self._create_or_match(nom, complement)

    def _create_or_match(self, nom, complement):
        for p in self.items:
            if key(p["nom"]) == key(nom) and (key(p["compl"]) == key(complement) or key(p["surnom"]) == key(complement)):
                return "found", p, []
        return self._create(nom, complement)

    def _create(self, nom, complement):
        pid = self._new_id()
        person = {"id": pid, "nom": nom, "compl": complement, "surnom": "", "c": self.iso(self.clock())}
        self.items.append(person)
        self._append(person)
        if self.activity:
            self.activity.log("person_add", id=pid, nom=self.activity.q(nom), compl=bool(complement))
        return "created", person, []

    def _new_id(self):
        used = {p["id"] for p in self.items}
        n = len(self.items) + 1
        while f"p{n}" in used:
            n += 1
        return f"p{n}"

    # ------------------------------------------------------------------ gestion (page de gestion)
    def add(self, nom, compl="", surnom=""):
        nom = _clean(nom)
        if not key(nom):
            return None
        with self.lock:
            person = {"id": self._new_id(), "nom": nom, "compl": _clean(compl), "surnom": _clean(surnom),
                      "c": self.iso(self.clock())}
            self.items.append(person)
            self._append(person)
            return person

    def update(self, pid, **changes):
        with self.lock:
            person = self.get(pid)
            if not person:
                return False
            for field in ("nom", "compl", "surnom"):
                if field in changes:
                    value = _clean(changes[field])
                    if field == "nom" and not key(value):
                        return False
                    person[field] = value
            self._rewrite()
            return True

    def remove(self, pid):
        with self.lock:
            before = len(self.items)
            self.items = [p for p in self.items if p["id"] != pid]
            if len(self.items) != before:
                self._rewrite()
                return True
            return False
