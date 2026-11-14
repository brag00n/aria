"""Personnes connues d'Aria (les interlocuteurs) : un fichier JSONL, ajout en fin de fichier, réécriture atomique pour modifier.

Fichier personnes.jsonl (dans le dossier de l'espace de noms), une personne par ligne :
    {"id":"p1","nom":"Olivier","compl":"Martin","surnom":"Oli","sexe":"m","c":"2026-10-30T14:02:11+00:00"}

Une personne peut être de type SYSTÈME (`type: "systeme"`, `cle` = identifiant interne d'un accès tiers) : une application ou un appareil (un robot)
qui utilise la mémoire. Absence de `type` = humain, donc les fichiers existants se lisent sans migration. Un système n'est pas un interlocuteur
d'Aria : il n'apparaît pas dans la liste déroulante (`listing`), mais il est une personne du foyer (graphe, page de gestion).

`nom` est le prénom (ou ce que la personne dit être son nom), `compl` un complément qui la distingue d'un homonyme (nom de famille),
`surnom` un nom d'usage, `sexe` "f", "m" ou vide (inconnu) : il sert au moteur de parenté (belle-fille, tante, grand-mère...). Module autonome : aucune importation depuis Aria, bibliothèque standard seulement.
"""
import json
import os
import threading
from pathlib import Path

import textproc
import audit_trace as trace_mod

MAX_FIELD = 60


def _clean(value):
    return " ".join(str(value or "").split())[:MAX_FIELD]


def _sexe(value):
    """'f', 'm' ou '' (inconnu) ; accepte aussi féminin/masculin/femme/homme."""
    v = key(str(value or ""))[:1]
    return v if v in ("f", "m") else ""


def est_systeme(person):
    """Vrai pour une personne de type système (accès tiers) ; absence de `type` = humain."""
    return bool(person) and person.get("type") == "systeme"


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
                                           "surnom": _clean(obj.get("surnom")), "sexe": _sexe(obj.get("sexe")), "c": obj.get("c", ""),
                                           **({"ref": True} if obj.get("ref") else {}),
                                           **({"tr": trace_mod.clean(obj.get("tr"))} if trace_mod.clean(obj.get("tr")) else {}),
                                           **({"type": "systeme", "cle": str(obj.get("cle", ""))[:40]} if obj.get("type") == "systeme" else {})})
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

    def listing(self, avec_citees=False):
        """Personnes pour la liste déroulante : les interlocuteurs (qui ont parlé). Les personnes seulement citées dans un lien
        (`ref`) n'y figurent pas, sauf `avec_citees` (page de gestion). Les systèmes (accès tiers) n'y figurent jamais : ce ne sont
        pas des interlocuteurs d'Aria."""
        with self.lock:
            return sorted(({"id": p["id"], "label": self.label(p)} for p in self.items
                           if not est_systeme(p) and (avec_citees or not p.get("ref"))), key=lambda x: key(x["label"]))

    def systeme(self, cle):
        """Personne système liée à la clé interne `cle` (id d'un accès tiers), ou None."""
        return next((p for p in self.items if est_systeme(p) and p.get("cle") == cle), None)

    def ensure_system(self, cle, nom, trace=None):
        """Retrouve ou crée la personne système d'un accès tiers. L'identité suit la CLÉ INTERNE, pas le nom : renommer l'accès renomme la
        fiche (même id, mêmes souvenirs). Retourne la personne, ou None si la clé ou le nom est vide."""
        cle, nom = _clean(cle), _clean(nom)
        if not cle or not key(nom):
            return None
        with self.lock:
            person = self.systeme(cle)
            if person:
                if person["nom"] != nom:
                    person["nom"] = nom
                    self._rewrite()
                    if self.activity:
                        self.activity.log("system_renamed", id=person["id"])
                return person
            person = {"id": self._new_id(), "nom": nom, "compl": "", "surnom": "", "sexe": "", "c": self.iso(self.clock()),
                      "type": "systeme", "cle": cle, **self._tr(trace)}
            self.items.append(person)
            self._append(person)
            if self.activity:
                self.activity.log("system_add", id=person["id"], cle=cle)
            return person

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
    @staticmethod
    def _tr(trace):
        """`{"tr": …}` si la trace est valide, sinon rien : une personne n'a ce champ que si son échange d'origine est connu."""
        tr = trace_mod.clean(trace)
        return {"tr": tr} if tr else {}

    def resolve(self, nom, complement="", surnom="", trace=None):
        """Retourne (statut, personne, candidats). statut :
          'found'      une seule personne correspond : à activer
          'created'    personne inconnue : créée
          'ambiguous'  plusieurs personnes (ou un prénom déjà pris sans complément) : on n'active rien
          'empty'      nom vide
        Un complément qui ne correspond à aucune personne portant ce prénom crée un homonyme."""
        status, person, candidates = self._resolve(_clean(nom), _clean(complement), trace)
        if person and person.get("ref"):                        # une personne seulement citée prend la parole : elle devient interlocutrice
            del person["ref"]
            with self.lock:
                self._rewrite()
        # Le surnom d'usage (« appelle-moi Oli ») est retenu sur la personne désignée, sans écraser un surnom existant
        surnom = _clean(surnom)
        if person and surnom and not person["surnom"] and key(surnom) != key(person["nom"]):
            person["surnom"] = surnom
            with self.lock:
                self._rewrite()
            if self.activity:
                self.activity.log("person_nickname", id=person["id"])
        return status, person, candidates

    def _reprendre_citee(self, nom, complement):
        """Une personne seulement CITÉE dans un lien (`ref`) prend la parole : on reprend sa fiche au lieu d'en créer une autre.
        Elle doit être la seule de ce prénom, et son nom de famille (s'il est connu) ne doit pas contredire celui qu'on donne."""
        if not complement and " " in nom:
            nom, complement = nom.split(" ", 1)
        candidates = [p for p in self.items if key(p["nom"]) == key(nom)]
        if len(candidates) != 1 or not candidates[0].get("ref"):
            return None
        person = candidates[0]
        if complement and person["compl"] and key(complement) != key(person["compl"]):
            return None
        if complement and not person["compl"]:
            person["compl"] = _clean(complement)
            with self.lock:
                self._rewrite()
        return person

    def _resolve(self, nom, complement, trace=None):
        if not key(nom):
            return "empty", None, []
        with self.lock:
            citee = self._reprendre_citee(nom, complement)
            if citee:
                return "found", citee, []
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
                    return self._create_or_match(parts[0], parts[1], trace)
                return self._create(nom, "", trace=trace)
            return self._create_or_match(nom, complement, trace)

    def _create_or_match(self, nom, complement, trace=None):
        for p in self.items:
            if key(p["nom"]) == key(nom) and (key(p["compl"]) == key(complement) or key(p["surnom"]) == key(complement)):
                return "found", p, []
        return self._create(nom, complement, trace=trace)

    def _create(self, nom, complement, ref=False, sexe="", trace=None):
        pid = self._new_id()
        person = {"id": pid, "nom": nom, "compl": complement, "surnom": "", "sexe": _sexe(sexe), "c": self.iso(self.clock()), **self._tr(trace)}
        if ref:
            person["ref"] = True
        self.items.append(person)
        self._append(person)
        if self.activity:
            self.activity.log("person_add", id=pid, nom=self.activity.q(nom), compl=bool(complement), ref=ref)
        return "created", person, []

    def reference(self, name, sexe="", trace=None):
        """Fiche d'une personne CITÉE dans un lien (« Emma » dans « Tiphaine est la mère de Emma »), créée si elle est inconnue.
        Elle est marquée « citée » (`ref`) : elle n'apparaît pas dans la liste des interlocuteurs tant qu'elle n'a pas parlé. Le sexe
        n'est renseigné que s'il est encore inconnu (une valeur saisie à la main n'est jamais écrasée). Retourne la personne, ou None
        si le nom est vide ou ambigu (plusieurs personnes le portent : on ne devine pas)."""
        name, sexe = _clean(name), _sexe(sexe)
        if not key(name):
            return None
        with self.lock:
            found = self.find(name)
            if len(found) > 1:
                return None
            if len(found) == 1:
                return self._renseigner(found[0], sexe=sexe)
            parts = name.split(" ", 1)
            if len(parts) == 2:                                 # « Marion Cousinier » : prénom + nom de famille
                same = [p for p in self.items if key(p["nom"]) == key(parts[0])]
                if len(same) == 1 and not same[0]["compl"]:     # « Marion » était connue sans nom de famille : on la complète
                    return self._renseigner(same[0], sexe=sexe, compl=_clean(parts[1]))
                return self._create(parts[0], _clean(parts[1]), ref=True, sexe=sexe, trace=trace)[1]
            return self._create(name, "", ref=True, sexe=sexe, trace=trace)[1]

    def _renseigner(self, person, sexe="", compl=""):
        changed = False
        if sexe and not person.get("sexe"):
            person["sexe"] = sexe
            changed = True
        if compl and not person["compl"]:
            person["compl"] = compl
            changed = True
        if changed:
            self._rewrite()
        return person

    def _new_id(self):
        used = {p["id"] for p in self.items}
        n = len(self.items) + 1
        while f"p{n}" in used:
            n += 1
        return f"p{n}"

    # ------------------------------------------------------------------ gestion (page de gestion)
    def add(self, nom, compl="", surnom="", sexe=""):
        nom = _clean(nom)
        if not key(nom):
            return None
        with self.lock:
            person = {"id": self._new_id(), "nom": nom, "compl": _clean(compl), "surnom": _clean(surnom),
                      "sexe": _sexe(sexe), "c": self.iso(self.clock())}
            self.items.append(person)
            self._append(person)
            return person

    def update(self, pid, **changes):
        with self.lock:
            person = self.get(pid)
            if not person:
                return False
            if "sexe" in changes:
                person["sexe"] = _sexe(changes["sexe"])
            if "ref" in changes:                       # True = seulement citée (hors de la liste des interlocuteurs), False = interlocutrice
                if changes["ref"]:
                    person["ref"] = True
                else:
                    person.pop("ref", None)
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
