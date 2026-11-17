"""Tests de la page de gestion (API sur un port libre) et du script d'analyse."""
import http.client
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import analyze  # noqa: E402
import store as store_mod  # noqa: E402
import ui  # noqa: E402
from activity import Activity  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UiCase(unittest.TestCase):
    token = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memui_"))
        self.activity = Activity(self.tmp / "logs")
        self.store = store_mod.Store(self.tmp, "default", activity=self.activity, lock_instance=False)
        self.store.active = "p0"
        self.port = free_port()
        self.httpd = ui.start(self.store, self.activity, self.port, self.token)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdr = {"Host": host or f"127.0.0.1:{self.port}"}
        hdr.update(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            hdr.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdr)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw


class UiTests(UiCase):
    def test_page_servie(self):
        status, raw = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"textContent" if False else b"<title>M", raw)

    def test_la_page_n_utilise_pas_innerhtml(self):
        self.assertNotIn(b"innerHTML", ui.PAGE.encode("utf-8"))

    def test_ajout_liste_modification_oubli_restauration(self):
        s, r = self.call("POST", "/api/add", {"texte": "Marion préfère 19 degrés", "type": "preference", "tags": "famille"})
        self.assertEqual((s, r["statut"]), (200, "added"))
        rid = r["id"]
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual(rows[0]["texte"], "Marion préfère 19 degrés")
        self.assertEqual(rows[0]["src"], "ui")
        self.assertEqual(rows[0]["etat"], "actif")
        self.assertTrue(self.call("POST", "/api/update", {"id": rid, "texte": "Marion préfère 20 degrés"})[1]["ok"])
        _, found = self.call("GET", "/api/memories?q=" + "20+degr%C3%A9s")
        self.assertEqual(found[0]["id"], rid)
        self.assertTrue(self.call("POST", "/api/pin", {"id": rid})[1]["ok"])
        self.assertEqual(self.call("GET", "/api/memories")[1][0]["etat"], "protege")
        self.assertTrue(self.call("POST", "/api/delete", {"id": rid})[1]["ok"])
        self.assertEqual(self.call("GET", "/api/memories")[1], [])
        self.assertEqual(self.call("GET", "/api/archive")[1][0]["why"], "ui")
        self.assertTrue(self.call("POST", "/api/restore", {"id": rid})[1]["ok"])
        self.assertEqual(len(self.call("GET", "/api/memories")[1]), 1)

    def test_stats_et_journal(self):
        self.call("POST", "/api/add", {"texte": "un souvenir de test"})
        self.assertEqual(self.call("GET", "/api/stats")[1]["total"], 1)
        events = [e["ev"] for e in self.call("GET", "/api/log")[1]]
        self.assertIn("ui", events)

    def test_hote_non_autorise_refuse(self):
        status, _ = self.call("GET", "/api/memories", host="evil.example.com")
        self.assertEqual(status, 403)

    def test_post_avec_mauvaise_origine_refuse(self):
        try:
            status, _ = self.call("POST", "/api/add", {"texte": "x y z"}, headers={"Origin": "http://evil.example.com"})
            self.assertEqual(status, 403)
        except (ConnectionError, http.client.HTTPException):
            pass        # refus avant lecture du corps : le client peut voir la connexion se fermer
        self.assertEqual(len(self.store.records), 0)

    def test_post_sans_json_refuse(self):
        try:
            status, _ = self.call("POST", "/api/add", b"texte=abc", headers={"Content-Type": "text/plain"})
            self.assertEqual(status, 415)
        except (ConnectionError, http.client.HTTPException):
            pass        # le serveur refuse sans lire le corps puis coupe : le client peut voir la connexion se fermer
        self.assertEqual(len(self.store.records), 0)        # dans tous les cas, rien n'a été enregistré

    def test_corps_trop_gros_refuse(self):
        try:
            status, _ = self.call("POST", "/api/add", {"texte": "a" * 50_000})
            self.assertEqual(status, 400)
        except (ConnectionError, http.client.HTTPException):
            pass        # le serveur a refusé le corps sans le lire puis coupé : le client peut voir la connexion se fermer
        self.assertEqual(len(self.store.records), 0)        # dans tous les cas, rien n'a été enregistré

    def test_texte_html_stocke_tel_quel_sans_etre_interprete(self):
        evil = "<script>alert(1)</script> chatte"
        self.call("POST", "/api/add", {"texte": evil})
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual(rows[0]["texte"], evil)        # JSON pur ; la page l'affiche avec textContent
        self.assertNotIn(b"innerHTML", ui.PAGE.encode("utf-8"))

    def test_route_inconnue(self):
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)


class UiPersonnesTests(UiCase):
    def add_person(self, nom, compl=""):
        return self.call("POST", "/api/personne/add", {"nom": nom, "compl": compl})[1]["id"]

    def test_personnes_et_foyer_dans_la_liste(self):
        a = self.add_person("Olivier", "Martin")
        self.call("POST", "/api/add", {"texte": "Aime le café serré", "type": "preference", "o": a})
        self.call("POST", "/api/add", {"texte": "Le portail est vert", "type": "foyer"})
        _, rows = self.call("GET", "/api/personnes")
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id[a]["souvenirs"], 1)
        self.assertEqual(by_id[""]["souvenirs"], 1)
        self.assertTrue(by_id[""]["special"])

    def test_la_page_peut_ajouter_pour_une_personne_sans_interlocuteur_actif(self):
        a = self.add_person("Olivier")
        self.store.active = ""
        status, r = self.call("POST", "/api/add", {"texte": "Aime le thé", "type": "preference", "o": a})
        self.assertEqual(r["statut"], "added")
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual((rows[0]["o"], rows[0]["proprietaire"]), (a, "Olivier"))

    def test_ajout_pour_une_personne_inconnue_refuse(self):
        _, r = self.call("POST", "/api/add", {"texte": "Aime le thé", "type": "preference", "o": "p99"})
        self.assertEqual(r["statut"], "personne_inconnue")
        self.assertEqual(len(self.store.records), 0)

    def test_attribuer_un_ancien_souvenir(self):
        a = self.add_person("Tiphaine")
        _, r = self.call("POST", "/api/add", {"texte": "Aime 21 degrés", "type": "foyer"})
        rid = r["id"]
        self.call("POST", "/api/update", {"id": rid, "type": "preference"})            # le foyer devient personnel...
        self.assertTrue(self.call("POST", "/api/update", {"id": rid, "o": a})[1]["ok"])     # ... et est attribué
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual(rows[0]["proprietaire"], "Tiphaine")
        self.assertFalse(self.call("POST", "/api/update", {"id": rid, "o": "p99"})[1]["ok"])

    def test_la_page_voit_tout_dans_la_recherche(self):
        a = self.add_person("Olivier")
        self.call("POST", "/api/add", {"texte": "Aime le café serré", "type": "preference", "o": a})
        self.store.active = ""
        _, rows = self.call("GET", "/api/memories?q=caf%C3%A9")
        self.assertEqual(len(rows), 1)

    def test_fusion_via_la_page(self):
        a, b = self.add_person("Olivier", "Martin"), self.add_person("Olivier", "Dupont")
        self.call("POST", "/api/add", {"texte": "Aime le thé", "type": "preference", "o": b})
        _, r = self.call("POST", "/api/personne/merge", {"id": b, "vers": a})
        self.assertEqual((r["ok"], r["deplaces"]), (True, 1))
        self.assertEqual([p["id"] for p in self.call("GET", "/api/personnes")[1] if not p.get("special")], [a])

    def test_supprimer_une_personne_avec_souvenirs_refuse(self):
        a = self.add_person("Olivier")
        self.call("POST", "/api/add", {"texte": "Aime le thé", "type": "preference", "o": a})
        _, r = self.call("POST", "/api/personne/delete", {"id": a})
        self.assertFalse(r["ok"])
        self.assertIn("souvenirs", r["erreur"])

    def test_renommer_une_personne(self):
        a = self.add_person("Olivier")
        self.assertTrue(self.call("POST", "/api/personne/update", {"id": a, "nom": "Olivier", "compl": "Cousinier", "surnom": "Oli"})[1]["ok"])
        person = next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == a)
        self.assertEqual((person["compl"], person["surnom"]), ("Cousinier", "Oli"))

    def test_relations(self):
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "femme de", "moi")
        _, rows = self.call("GET", "/api/relations")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["rel"]["a"], rows[0]["rel"]["l"]), ("Tiphaine", "femme de"))
        self.assertEqual(rows[0]["proprietaire"], "Foyer (partagé)")

    def test_la_page_peut_oublier_le_souvenir_d_un_autre(self):
        a = self.add_person("Olivier")
        _, r = self.call("POST", "/api/add", {"texte": "Aime le thé", "type": "preference", "o": a})
        self.store.active = ""
        self.assertTrue(self.call("POST", "/api/delete", {"id": r["id"]})[1]["ok"])


class UiEditionTests(UiCase):
    def make_relation(self):
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "femme de", "moi")
        return next(r for r in self.store.records.values() if r["t"] == "relation")

    def test_modifier_le_renfort(self):
        owner = self.store.personnes.add("Olivier")["id"]
        _, r = self.call("POST", "/api/add", {"texte": "Aime le café serré", "type": "preference", "o": owner})
        rid = r["id"]
        self.assertTrue(self.call("POST", "/api/update", {"id": rid, "s": "1.5"})[1]["ok"])
        self.assertEqual(self.store.records[rid]["s"], 1.5)
        self.call("POST", "/api/update", {"id": rid, "s": "99"})
        self.assertEqual(self.store.records[rid]["s"], 2.0)             # borné
        self.call("POST", "/api/update", {"id": rid, "s": "n'importe quoi"})
        self.assertEqual(self.store.records[rid]["s"], 2.0)             # valeur invalide ignorée

    def test_editer_une_relation_recalcule_le_texte_et_l_index(self):
        rel = self.make_relation()
        rid = rel["id"]
        ok = self.call("POST", "/api/update", {"id": rid, "rel": {"a": "Tiphaine", "l": "soeur de", "b": "Olivier"}})[1]
        self.assertTrue(ok["ok"])
        self.assertEqual(self.store.records[rid]["txt"], "Tiphaine est la soeur de Olivier")
        self.assertEqual(self.store.records[rid]["rel"]["l"], "soeur de")
        self.assertEqual(self.store.search("frere", touch=False, everything=True)[0][0]["id"], rid)    # nouvel index : fratrie

    def test_relation_a_deux_personnes_differentes(self):
        rid = self.make_relation()["id"]
        x = self.call("POST", "/api/update", {"id": rid, "rel": {"a": "Olivier", "l": "ami de", "b": "olivier"}})[1]
        self.assertFalse(x["ok"])
        self.assertIn("deux personnes", x["erreur"])
        x = self.call("POST", "/api/update", {"id": rid, "rel": {"a": "", "l": "ami de", "b": "Paul"}})[1]
        self.assertFalse(x["ok"])

    def test_rel_refuse_sur_un_souvenir_ordinaire(self):
        _, r = self.call("POST", "/api/add", {"texte": "Aime le café serré", "type": "preference", "o": self.store.personnes.add("Olivier")["id"]})
        x = self.call("POST", "/api/update", {"id": r["id"], "rel": {"a": "A", "l": "ami de", "b": "B"}})[1]
        self.assertFalse(x["ok"])
        self.assertIn("pas une relation", x["erreur"])

    def test_la_relation_editee_survit_au_rechargement(self):
        rid = self.make_relation()["id"]
        self.call("POST", "/api/update", {"id": rid, "rel": {"a": "Tiphaine", "l": "soeur de", "b": "Paul"}, "s": "1.3"})
        self.store.close()
        import store as store_mod2
        again = store_mod2.Store(self.tmp, "default", lock_instance=False)
        rec = again.records[rid]
        self.assertEqual((rec["rel"]["b"], rec["s"]), ("Paul", 1.3))
        self.assertEqual(rec["txt"], "Tiphaine est la soeur de Paul")

    def test_changer_le_type_et_le_proprietaire_en_une_fois(self):
        a = self.store.personnes.add("Olivier")["id"]
        _, r = self.call("POST", "/api/add", {"texte": "Le portail est vert", "type": "foyer"})
        x = self.call("POST", "/api/update", {"id": r["id"], "type": "fait", "o": a, "tags": "maison,pinned", "imp": "7"})[1]
        self.assertTrue(x["ok"])
        rec = self.store.records[r["id"]]
        self.assertEqual((rec["t"], rec["o"], rec["imp"]), ("fait", a, 7.0))
        self.assertIn("pinned", rec["tags"])


class UiSexeTests(UiCase):
    def test_modifier_le_sexe_d_une_personne(self):
        pid = self.call("POST", "/api/personne/add", {"nom": "Emma"})[1]["id"]
        self.assertTrue(self.call("POST", "/api/personne/update", {"id": pid, "sexe": "f"})[1]["ok"])
        person = next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == pid)
        self.assertEqual(person["sexe"], "f")
        self.call("POST", "/api/personne/update", {"id": pid, "sexe": "zzz"})
        person = next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == pid)
        self.assertEqual(person["sexe"], "")                 # valeur invalide : sexe inconnu, jamais un texte libre

    def test_ajouter_une_personne_avec_son_sexe(self):
        pid = self.call("POST", "/api/personne/add", {"nom": "Paul", "sexe": "masculin"})[1]["id"]
        person = next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == pid)
        self.assertEqual(person["sexe"], "m")

    def test_connue_d_aria_ajoute_ou_retire_de_la_liste_des_interlocuteurs(self):
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "conjoint", "moi")
        tiphaine = self.store.personnes.find("Tiphaine")[0]["id"]
        liste = lambda: {p["label"] for p in self.store.personnes.listing()}
        self.assertNotIn("Tiphaine", liste())                                         # seulement citée
        self.assertTrue(self.call("POST", "/api/personne/update", {"id": tiphaine, "connue": True})[1]["ok"])
        self.assertIn("Tiphaine", liste())
        row = next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == tiphaine)
        self.assertFalse(row["citee"])
        self.assertTrue(self.call("POST", "/api/personne/update", {"id": tiphaine, "connue": False})[1]["ok"])
        self.assertNotIn("Tiphaine", liste())
        self.assertTrue(next(p for p in self.call("GET", "/api/personnes")[1] if p["id"] == tiphaine)["citee"])
        # ses liens et ses fiches restent intacts
        self.assertTrue(i.graphe(tiphaine)["voisins"])

    def test_on_ne_retire_pas_l_interlocuteur_actif(self):
        pid = self.call("POST", "/api/personne/add", {"nom": "Paul"})[1]["id"]
        self.store.active = pid
        x = self.call("POST", "/api/personne/update", {"id": pid, "connue": False})[1]
        self.assertFalse(x["ok"])
        self.assertIn("actif", x["erreur"])
        self.assertIn("Paul", {p["label"] for p in self.store.personnes.listing()})

    def test_modifier_sans_toucher_au_statut_le_conserve(self):
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "conjoint", "moi")
        tiphaine = self.store.personnes.find("Tiphaine")[0]["id"]
        self.call("POST", "/api/personne/update", {"id": tiphaine, "surnom": "Titi"})
        self.assertTrue(self.store.personnes.get(tiphaine).get("ref"))

    def test_les_relations_ne_sont_plus_listees_dans_les_souvenirs(self):
        self.assertIn("Ouvrir Relations", ui.PAGE)                       # renvoi vers l'onglet Relations
        self.assertIn("r.type!=='relation'", ui.PAGE)                    # masquées par défaut dans l'onglet Souvenirs
        status, rows = self.call("GET", "/api/relations")                # toujours servies par leur propre onglet
        self.assertEqual(status, 200)
        self.assertTrue(all(r["type"] == "relation" for r in rows))

    def test_recherche_dans_les_onglets_personnes_et_relations(self):
        for mot in ("Rechercher une personne", "Rechercher une relation", "const match="):
            self.assertIn(mot, ui.PAGE)

    def test_la_page_propose_connue_par_aria(self):
        self.assertIn("Connue par Aria", ui.PAGE)

    def test_les_personnes_citees_sont_signalees(self):
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "femme de", "moi")
        rows = {p["nom"]: p for p in self.call("GET", "/api/personnes")[1] if not p.get("special")}
        self.assertTrue(rows["Tiphaine"]["citee"])
        self.assertFalse(rows["Olivier"]["citee"])
        self.assertEqual(rows["Tiphaine"]["sexe"], "f")


class UiGrapheTests(UiCase):
    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        i = Interlocuteurs(self.store)
        i.identifier("Olivier")
        i.relier("Tiphaine", "femme de", "moi")
        i.relier("Lucy", "amie de", "moi")
        self.ids = {p["nom"]: p["id"] for p in self.store.personnes.items}

    def test_api_graphe(self):
        status, g = self.call("GET", "/api/graphe?id=" + self.ids["Olivier"])
        self.assertEqual(status, 200)
        self.assertEqual(g["centre"]["nom"], "Olivier")
        v = {x["nom"]: x for x in g["voisins"]}
        self.assertEqual(v["Tiphaine"]["categorie"], "famille")
        self.assertEqual(v["Lucy"]["categorie"], "connaissance")

    def test_api_graphe_recentre(self):
        g = self.call("GET", "/api/graphe?id=" + self.ids["Tiphaine"])[1]
        self.assertEqual(g["centre"]["nom"], "Tiphaine")
        self.assertIn("Olivier", {x["nom"] for x in g["voisins"]})

    def test_api_graphe_niveau(self):
        self.store.personnes.add("Paul")
        from interlocuteur import Interlocuteurs
        Interlocuteurs(self.store).relier("Paul", "ami de", "Lucy")
        n1 = self.call("GET", "/api/graphe?id=" + self.ids["Olivier"])[1]
        n2 = self.call("GET", "/api/graphe?id=" + self.ids["Olivier"] + "&niveau=2")[1]
        self.assertEqual((n1["niveau"], n2["niveau"]), (1, 2))
        self.assertNotIn("Paul", {v["nom"] for v in n1["voisins"]})
        self.assertIn("Paul", {v["nom"] for v in n2["voisins"]})
        self.assertEqual(self.call("GET", "/api/graphe?id=" + self.ids["Olivier"] + "&niveau=42")[1]["niveau"], 5)

    def test_renommer_une_personne_conserve_ses_liens(self):
        self.call("POST", "/api/personne/update", {"id": self.ids["Lucy"], "nom": "Clémence"})
        g = self.call("GET", "/api/graphe?id=" + self.ids["Olivier"] + "&niveau=2")[1]
        self.assertIn("Clémence", {x["nom"] for x in g["voisins"]})
        rel = next(r for r in self.store.records.values() if r["t"] == "relation" and "Clémence" in r["txt"])
        self.assertEqual(rel["rel"]["a"], "Clémence")

    def test_api_creer_une_relation(self):
        tiphaine, olivier = self.ids["Tiphaine"], self.ids["Olivier"]
        paul = self.call("POST", "/api/personne/add", {"nom": "Paul"})[1]["id"]
        x = self.call("POST", "/api/relation/add", {"a": paul, "lien": "enfant", "b": tiphaine})[1]
        self.assertTrue(x["ok"])
        rows = self.call("GET", "/api/relations")[1]
        self.assertIn(x["id"], {r["id"] for r in rows})
        self.assertEqual(next(r for r in rows if r["id"] == x["id"])["rel"]["t"], "enfant")
        g = self.call("GET", "/api/graphe?id=" + tiphaine)[1]
        self.assertIn("Paul", {v["nom"] for v in g["voisins"]})

    def test_api_creer_une_relation_refus(self):
        tiphaine, olivier = self.ids["Tiphaine"], self.ids["Olivier"]
        for corps in ({"a": tiphaine, "lien": "conjoint", "b": tiphaine}, {"a": tiphaine, "lien": "zzz", "b": olivier},
                      {"a": "", "lien": "ami", "b": olivier}, {"a": tiphaine, "lien": "autre", "b": olivier},
                      {"a": tiphaine, "lien": "conjoint", "b": olivier}):                  # ce dernier existe déjà
            x = self.call("POST", "/api/relation/add", corps)[1]
            self.assertFalse(x["ok"], corps)
            self.assertTrue(x["erreur"])

    def test_la_page_propose_filtres_et_creation_de_relation(self):
        for mot in ("Nouvelle relation", "Dans : tous les champs", "Sexe : tous", "Statut : tous", "Lien : tous", "/api/relation/add"):
            self.assertIn(mot, ui.PAGE)
        creation = ui.PAGE[ui.PAGE.index("ntype=el('select'"):][:200]
        self.assertNotIn("relation", creation)                                             # plus de type « relation » à la création d'un souvenir

    def test_api_graphe_inconnu(self):
        status, g = self.call("GET", "/api/graphe?id=p404")
        self.assertEqual(status, 404)
        self.assertIn("erreur", g)
        self.assertEqual(self.call("GET", "/api/graphe")[0], 404)

    def test_la_page_propose_le_graphe_colore(self):
        for mot in ("openGraph", "--fam", "--con"):
            self.assertIn(mot, ui.PAGE)


class UiSystemeTests(UiCase):
    """Personnes de type système (agent) dans la page de gestion."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        Interlocuteurs(self.store).identifier("Olivier", "Cousinier")
        self.olivier = self.store.active
        self.robot = self.store.personnes.ensure_system("k_robot", "Robot")["id"]
        from service import Service
        self.svc = Service(self.store, Interlocuteurs(self.store))
        self.svc.memoriser("La batterie est à 80 pour cent", principal=self.robot)

    def lignes(self):
        return {p["id"]: p for p in self.call("GET", "/api/personnes")[1] if not p.get("special")}

    def test_un_systeme_est_signale_avec_sa_cle_interne_et_ses_souvenirs(self):
        p = self.lignes()[self.robot]
        self.assertTrue(p["systeme"])
        self.assertEqual((p["cle"], p["souvenirs"]), ("k_robot", 1))
        self.assertFalse(self.lignes()[self.olivier]["systeme"])

    def test_la_page_voit_le_proprietaire_des_souvenirs_du_robot(self):
        rows = self.call("GET", "/api/memories?q=batterie")[1]
        self.assertEqual([(r["texte"], r["proprietaire"]) for r in rows], [("La batterie est à 80 pour cent", "Robot")])

    def test_un_systeme_ne_devient_pas_interlocuteur_d_aria(self):
        x = self.call("POST", "/api/personne/update", {"id": self.robot, "connue": True})[1]
        self.assertTrue(x["ok"])
        self.assertTrue(self.store.is_system(self.robot))
        self.assertNotIn(self.robot, {p["id"] for p in self.store.personnes.listing(avec_citees=True)})
        # l'interrupteur n'a rien changé, y compris dans le fichier : la fiche n'est ni « citée » ni marquée autrement
        fiche = self.store.personnes.get(self.robot)
        self.assertNotIn("ref", fiche)
        brut = [json.loads(l) for l in (self.tmp / "default" / "personnes.jsonl").read_text("utf-8").splitlines()]
        self.assertEqual([{k: v for k, v in p.items() if k in ("type", "cle", "ref")} for p in brut if p["id"] == self.robot],
                         [{"type": "systeme", "cle": "k_robot"}])
        self.call("POST", "/api/personne/update", {"id": self.robot, "connue": False})            # même décoché : pas de « citée » sur un système
        self.assertNotIn("ref", self.store.personnes.get(self.robot))

    def test_on_ne_fusionne_pas_un_systeme_avec_une_personne(self):
        for source, cible in ((self.robot, self.olivier), (self.olivier, self.robot)):
            x = self.call("POST", "/api/personne/merge", {"id": source, "vers": cible})[1]
            self.assertFalse(x["ok"])
            self.assertIn("séparées", x["erreur"])
        self.assertIsNotNone(self.store.personnes.get(self.robot))
        self.assertIsNotNone(self.store.personnes.get(self.olivier))
        self.assertEqual([r["o"] for r in self.store.records.values() if "batterie" in r["txt"]], [self.robot])

    def test_fusionner_deux_systemes_reste_possible(self):
        autre = self.store.personnes.ensure_system("k_robot2", "Robot bis")["id"]
        x = self.call("POST", "/api/personne/merge", {"id": autre, "vers": self.robot})[1]
        self.assertTrue(x["ok"])

    def test_la_page_affiche_le_badge_systeme(self):
        self.assertIn("système", ui.PAGE)
        self.assertIn("Agents (système)", ui.PAGE)


class UiTokenTests(UiCase):
    token = "secret-test"

    def test_jeton_exige_pour_l_api(self):
        self.assertEqual(self.call("GET", "/api/memories")[0], 401)
        self.assertEqual(self.call("GET", "/api/memories", headers={"X-Token": "secret-test"})[0], 200)
        self.assertEqual(self.call("GET", "/")[0], 200)       # la page elle-même se charge, ses appels portent le jeton


class AnalyzeTests(unittest.TestCase):
    def test_resume(self):
        tmp = Path(tempfile.mkdtemp(prefix="memana_"))
        try:
            act = Activity(tmp / "logs")
            s = store_mod.Store(tmp, "default", activity=act, lock_instance=False)
            s.active = "p0"
            _, a = s.add("Marion préfère 19 degrés", "préférence")
            s.add("Le compteur Linky est l'équipement 270")
            s.add("Marion préfère 19 degrés")
            s.search("Marion")
            s.search("zzzzqqq")
            summary = analyze.summarize(tmp)
            self.assertEqual(summary["souvenirs"], 2)
            self.assertEqual(summary["ajouts"], 3)
            self.assertEqual(summary["doublons"], 1)
            self.assertEqual(summary["recherches_llm"], 2)
            self.assertEqual(summary["plus_rappeles"][0][0], a["id"])
            self.assertEqual(len(summary["jamais_rappeles"]), 1)
            self.assertEqual(len(summary["recherches_en_repli"]), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dossier_vide(self):
        tmp = Path(tempfile.mkdtemp(prefix="memana_"))
        try:
            summary = analyze.summarize(tmp)
            self.assertEqual(summary["souvenirs"], 0)
            self.assertEqual(summary["recherches_llm"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
