"""Tests des interlocuteurs : personnes, propriétaire des souvenirs, visibilité, homonymes, relations."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring  # noqa: E402
import store as store_mod  # noqa: E402
from personnes import Personnes  # noqa: E402


class Clock:
    def __init__(self, start=1_800_000_000.0):
        self.t = start

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="meminterl_"))
        self.clock = Clock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, active=""):
        s = store_mod.Store(self.tmp, "default", clock=self.clock, lock_instance=False)
        s.active = active
        return s


class PersonnesTests(Base):
    def persons(self):
        return Personnes(self.tmp, store_mod.iso, self.clock)

    def test_creation_puis_retrouve(self):
        p = self.persons()
        status, olivier, _ = p.resolve("Olivier")
        self.assertEqual(status, "created")
        status, again, _ = p.resolve("olivier")
        self.assertEqual((status, again["id"]), ("found", olivier["id"]))

    def test_accents_et_casse_ignores(self):
        p = self.persons()
        p.resolve("Éric")
        self.assertEqual(p.resolve("eric")[0], "found")
        self.assertEqual(p.resolve("ÉRIC")[0], "found")

    def test_un_seul_olivier_connu_est_retrouve_par_son_prenom(self):
        p = self.persons()
        _, martin, _ = p.resolve("Olivier", "Martin")
        status, person, _ = p.resolve("Olivier")
        self.assertEqual((status, person["id"]), ("found", martin["id"]))

    def test_deux_homonymes_sans_complement_est_ambigu(self):
        p = self.persons()
        p.resolve("Olivier", "Martin")
        p.resolve("Olivier", "Dupont")
        status, person, candidates = p.resolve("Olivier")
        self.assertEqual(status, "ambiguous")
        self.assertIsNone(person)
        self.assertEqual(len(candidates), 2)

    def test_complement_cree_un_homonyme(self):
        p = self.persons()
        _, a, _ = p.resolve("Olivier", "Martin")
        status, b, _ = p.resolve("Olivier", "Dupont")
        self.assertEqual(status, "created")
        self.assertNotEqual(a["id"], b["id"])
        labels = {x["label"] for x in p.listing()}
        self.assertEqual(labels, {"Olivier (Martin)", "Olivier (Dupont)"})

    def test_complement_retrouve_la_bonne_personne(self):
        p = self.persons()
        _, a, _ = p.resolve("Olivier", "Martin")
        p.resolve("Olivier", "Dupont")
        self.assertEqual(p.resolve("Olivier", "Martin")[1]["id"], a["id"])
        self.assertEqual(p.resolve("Olivier Martin")[1]["id"], a["id"])

    def test_nom_complet_donne_d_un_coup(self):
        p = self.persons()
        status, person, _ = p.resolve("Tiphaine Durand")
        self.assertEqual(status, "created")
        self.assertEqual((person["nom"], person["compl"]), ("Tiphaine", "Durand"))

    def test_surnom(self):
        p = self.persons()
        _, person, _ = p.resolve("Olivier", "Martin")
        p.update(person["id"], surnom="Oli")
        self.assertEqual(p.resolve("Oli")[1]["id"], person["id"])

    def test_nom_vide(self):
        self.assertEqual(self.persons().resolve("   ")[0], "empty")

    def test_persistance(self):
        p = self.persons()
        p.resolve("Olivier", "Martin")
        p.resolve("Tiphaine")
        again = self.persons()
        self.assertEqual(len(again.items), 2)
        self.assertEqual(again.resolve("tiphaine")[0], "found")

    def test_renommer_et_supprimer(self):
        p = self.persons()
        _, a, _ = p.resolve("Olivier")
        self.assertTrue(p.update(a["id"], nom="Olivier Jr"))
        self.assertEqual(self.persons().get(a["id"])["nom"], "Olivier Jr")
        self.assertTrue(p.remove(a["id"]))
        self.assertEqual(self.persons().items, [])

    def test_ligne_corrompue_ignoree(self):
        p = self.persons()
        p.resolve("Olivier")
        with open(self.tmp / "personnes.jsonl", "ab") as f:
            f.write(b"pas du json\n")
        self.assertEqual(len(self.persons().items), 1)


class OwnerTests(Base):
    def test_inconnue_ne_memorise_rien(self):
        s = self.make(active="")
        status, rec = s.add("J'aime le café serré", "préférence")
        self.assertEqual((status, rec), ("unknown_speaker", None))
        self.assertEqual(len(s.records), 0)

    def test_inconnue_ne_memorise_pas_non_plus_un_fait_du_foyer(self):
        s = self.make(active="")
        self.assertEqual(s.add("Le portail est vert", "foyer")[0], "unknown_speaker")

    def test_souvenir_prive_attribue_a_l_actif(self):
        s = self.make(active="p1")
        _, rec = s.add("J'aime le café serré", "préférence")
        self.assertEqual(rec["o"], "p1")

    def test_prive_invisible_pour_les_autres_et_pour_inconnue(self):
        s = self.make(active="p1")
        s.add("J'aime le café serré", "préférence")
        s.active = "p2"
        self.assertEqual(s.search("café", touch=False)[0], [])
        s.active = ""
        self.assertEqual(s.search("café", touch=False)[0], [])
        s.active = "p1"
        self.assertEqual(len(s.search("café", touch=False)[0]), 1)

    def test_le_repli_ne_revele_pas_les_souvenirs_des_autres(self):
        s = self.make(active="p1")
        s.add("Mon code secret est dans le tiroir", "fait")
        s.active = "p2"
        res, fallback = s.search("zzzzqqq", touch=False)
        self.assertEqual(res, [])

    def test_partage_visible_de_tous(self):
        s = self.make(active="p1")
        s.add("Le portail est vert", "foyer")
        for who in ("p1", "p2", ""):
            s.active = who
            self.assertEqual(len(s.search("portail", touch=False)[0]), 1, who)

    def test_deduplication_par_proprietaire(self):
        s = self.make(active="p1")
        a = s.add("J'aime le café serré", "préférence")
        s.active = "p2"
        b = s.add("J'aime le café serré", "préférence")
        self.assertEqual((a[0], b[0]), ("added", "added"))
        self.assertEqual(len(s.records), 2)
        self.assertEqual(s.add("J'aime le café serré", "préférence")[0], "duplicate")

    def test_ancien_hash_conserve_pour_le_partage(self):
        s = self.make(active="p1")
        _, rec = s.add("Le portail est vert", "foyer")
        self.assertEqual(rec["h"], store_mod.Store._hash("Le portail est vert"))

    def test_anciens_souvenirs_sans_proprietaire_sont_partages(self):
        s = self.make(active="p1")
        ev = {"op": "add", "ts": store_mod.iso(self.clock()), "r": {
            "id": "m_old001", "t": "fait", "txt": "La chatte s'appelle Réglisse", "imp": 5, "tags": [], "h": "",
            "c": store_mod.iso(self.clock()), "last": store_mod.iso(self.clock()), "hits": 0, "s": 1.0, "src": "llm"}}
        with open(s.ns_dir / "events.jsonl", "ab") as f:
            f.write((json.dumps(ev) + "\n").encode("utf-8"))
        for who in ("p1", "p2", ""):
            s.active = who
            self.assertEqual(len(s.search("chatte Réglisse", touch=False)[0]), 1, who)

    def test_oubli_des_souvenirs_d_autrui_refuse(self):
        s = self.make(active="p1")
        _, rec = s.add("J'aime le café serré", "préférence")
        s.active = "p2"
        self.assertFalse(s.delete(rec["id"]))
        s.active = ""
        self.assertFalse(s.delete(rec["id"]))
        s.active = "p1"
        self.assertTrue(s.delete(rec["id"]))

    def test_oubli_du_partage_exige_d_etre_identifie(self):
        s = self.make(active="p1")
        _, rec = s.add("Le portail est vert", "foyer")
        s.active = ""
        self.assertFalse(s.delete(rec["id"]))
        s.active = "p2"
        self.assertTrue(s.delete(rec["id"]))

    def test_gestion_peut_tout_voir_et_tout_oublier(self):
        s = self.make(active="p1")
        _, rec = s.add("J'aime le café serré", "préférence")
        s.active = ""
        self.assertEqual(len(s.search("café", touch=False, everything=True)[0]), 1)
        self.assertTrue(s.delete(rec["id"], force=True))

    def test_la_page_peut_attribuer_a_une_personne_meme_sans_actif(self):
        s = self.make(active="")
        status, rec = s.add("Aime le thé", "préférence", src="ui", owner="p3")
        self.assertEqual((status, rec["o"]), ("added", "p3"))

    def test_rappel_ne_renforce_que_les_souvenirs_visibles(self):
        s = self.make(active="p1")
        _, mine = s.add("J'aime le café serré", "préférence")
        s.active = "p2"
        _, other = s.add("J'aime le café au lait", "préférence")
        s.search("café")
        self.assertEqual((mine["hits"], other["hits"]), (0, 1))

    def test_rejeu_conserve_le_proprietaire(self):
        s = self.make(active="p1")
        _, rec = s.add("J'aime le café serré", "préférence")
        s.close()
        again = self.make(active="p1")
        self.assertEqual(again.records[rec["id"]]["o"], "p1")
        again.active = "p2"
        self.assertEqual(again.search("café", touch=False)[0], [])

    def test_compaction_conserve_le_proprietaire(self):
        s = self.make(active="p1")
        _, rec = s.add("J'aime le café serré", "préférence")
        s.compact()
        s.close()
        self.assertEqual(self.make(active="p1").records[rec["id"]]["o"], "p1")


class RelationTests(Base):
    def test_relation_jamais_oubliee(self):
        s = self.make(active="p1")
        _, rec = s.add("Tiphaine est la femme d'Olivier", "relation", rel={"a": "Tiphaine", "l": "femme de", "b": "Olivier"})
        self.assertEqual(rec["o"], "")
        self.assertIn("core", rec["tags"])
        self.clock.t += 3000 * 86400
        self.assertEqual(s.state(rec), "protege")
        s.compact()
        self.assertIn(rec["id"], s.records)

    def test_relation_visible_de_tous_et_conservee_au_rechargement(self):
        s = self.make(active="p1")
        _, rec = s.add("Tiphaine est la femme d'Olivier", "relation", rel={"a": "Tiphaine", "l": "femme de", "b": "Olivier"})
        s.close()
        again = self.make(active="p2")
        found = again.search("Tiphaine femme", touch=False)[0]
        self.assertEqual(found[0]["rel"]["l"], "femme de")


class OutilsTests(Base):
    """Les réponses des outils vus par le LLM et l'état lu par Aria (couche interlocuteur.py)."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    def test_etat_initial_inconnue(self):
        etat = self.i.etat("?")
        self.assertEqual(etat["actif"], "")
        self.assertEqual(etat["liste"], [])
        self.assertIn("Inconnue", etat["contexte"])
        self.assertIn("à qui ai-je l'honneur", etat["contexte"])

    def test_identifier_cree_et_active(self):
        reponse = self.i.identifier("Olivier")
        self.assertIn("créé et actif", reponse)
        self.assertIn("questions", reponse)
        etat = self.i.etat("?")
        self.assertEqual(etat["actif"], "p1")
        self.assertEqual(etat["liste"], [{"id": "p1", "label": "Olivier"}])
        self.assertIn("Olivier", etat["contexte"])
        self.assertNotIn("Inconnue", etat["contexte"])

    def test_memorisation_apres_identification_dans_le_meme_tour(self):
        self.i.identifier("Olivier")
        status, rec = self.s.add("Aime le café serré", "préférence")
        self.assertEqual((status, rec["o"]), ("added", "p1"))

    def test_memoriser_refuse_tant_qu_inconnue(self):
        status, _ = self.s.add("Aime le café serré", "préférence")
        self.assertEqual(status, "unknown_speaker")
        self.assertIn("Interlocuteur inconnu", self.i.refus(status))

    def test_homonyme_n_active_personne(self):
        self.i.identifier("Olivier", "Martin")
        self.i.identifier("Olivier", "Dupont")
        self.i.etat("")
        reponse = self.i.identifier("Olivier")
        self.assertIn("Plusieurs personnes", reponse)
        self.assertEqual(self.s.active, "")

    def test_activer_par_aria_et_identifiant_inconnu(self):
        self.i.identifier("Olivier")
        self.assertEqual(self.i.etat("")["actif"], "")
        self.assertEqual(self.i.etat("p1")["actif"], "p1")
        self.assertEqual(self.i.etat("p99")["actif"], "")          # identifiant inexistant : retombe sur Inconnue
        self.assertEqual(self.i.etat("?")["actif"], "")

    def test_changement_d_interlocuteur_signale(self):
        self.i.identifier("Olivier")
        self.assertIn("Changement d'interlocuteur", self.i.identifier("Tiphaine") + self.i.identifier("Olivier"))

    def test_relier_avec_moi(self):
        self.i.identifier("Olivier")
        reponse = self.i.relier("Tiphaine", "femme de", "moi")
        self.assertIn("Tiphaine est la femme de Olivier", reponse)
        self.assertTrue(self.i.relier("Tiphaine", "femme de", "moi").startswith("Déjà connu"))      # idempotent
        self.s.active = "p2"
        found = self.s.search("Tiphaine femme", touch=False)[0]
        self.assertEqual(found[0]["rel"]["l"], "femme de")

    def test_relier_refuse_si_inconnue(self):
        self.assertIn("Interlocuteur inconnu", self.i.relier("Tiphaine", "femme de", "Olivier"))

    def test_relier_valide_les_entrees(self):
        self.i.identifier("Olivier")
        self.assertIn("Il manque", self.i.relier("", "femme de", "moi"))
        self.assertIn("elle-même", self.i.relier("moi", "ami de", "Olivier"))

    def test_relier_renseigne_les_identifiants_connus(self):
        self.i.identifier("Olivier")
        self.i.identifier("Tiphaine")
        self.i.etat("p1")
        self.i.relier("Tiphaine", "femme de", "moi")
        rec = next(r for r in self.s.records.values() if r["t"] == "relation")
        self.assertEqual((rec["rel"]["ia"], rec["rel"]["ib"]), ("p2", "p1"))

    def test_changement_enregistre_dans_le_journal_d_activite(self):
        from activity import Activity
        act = Activity(self.tmp / "logs")
        s = store_mod.Store(self.tmp / "autre", "default", activity=act, clock=self.clock, lock_instance=False)
        from interlocuteur import Interlocuteurs
        Interlocuteurs(s).identifier("Olivier")
        events = [json.loads(l) for l in (self.tmp / "logs" / "activity.jsonl").read_text("utf-8").splitlines()]
        speakers = [e for e in events if e["ev"] == "speaker"]
        self.assertEqual((speakers[0]["frm"], speakers[0]["to"], speakers[0]["src"]), ("", "p1", "llm"))


class SurnomEtGestionTests(Base):
    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    def test_surnom_retenu_a_la_creation(self):
        self.i.identifier("Olivier Cousinier", surnom="Olivier")
        person = self.s.personnes.get("p1")
        self.assertEqual((person["nom"], person["compl"]), ("Olivier", "Cousinier"))
        self.assertEqual(self.s.personnes.resolve("Oli", surnom="")[0], "created")      # « Oli » n'est pas encore un surnom

    def test_surnom_different_du_prenom_est_enregistre_et_retrouve(self):
        self.i.identifier("Olivier", "Cousinier", surnom="Oli")
        self.assertEqual(self.s.personnes.get("p1")["surnom"], "Oli")
        self.i.etat("")
        self.assertIn("est maintenant actif", self.i.identifier("Oli"))
        self.assertEqual(self.s.active, "p1")

    def test_surnom_existant_non_ecrase(self):
        self.i.identifier("Olivier", "Cousinier", surnom="Oli")
        self.i.identifier("Olivier", "Cousinier", surnom="Livier")
        self.assertEqual(self.s.personnes.get("p1")["surnom"], "Oli")

    def test_rattacher_un_ancien_souvenir_a_une_personne(self):
        anonyme = self.make(active="p9")
        ev = {"op": "add", "ts": store_mod.iso(self.clock()), "r": {
            "id": "m_old002", "t": "fait", "txt": "Ma femme s'appelle Tiphaine", "imp": 5, "tags": [], "h": "",
            "c": store_mod.iso(self.clock()), "last": store_mod.iso(self.clock()), "hits": 0, "s": 1.0, "src": "llm"}}
        with open(anonyme.ns_dir / "events.jsonl", "ab") as f:
            f.write((json.dumps(ev) + "\n").encode("utf-8"))
        anonyme.active = "p2"
        self.assertEqual(len(anonyme.search("Tiphaine", touch=False)[0]), 1)             # partagé tant qu'il n'a pas de propriétaire
        self.assertTrue(anonyme.update("m_old002", {"o": "p1"}))
        self.assertEqual(anonyme.search("Tiphaine", touch=False)[0], [])                 # privé à p1 maintenant
        anonyme.active = "p1"
        self.assertEqual(len(anonyme.search("Tiphaine", touch=False)[0]), 1)
        anonyme.close()
        again = self.make(active="p2")
        self.assertEqual(again.records["m_old002"]["o"], "p1")                           # persiste après rechargement

    def test_changer_en_type_partage_retire_le_proprietaire(self):
        s = self.make(active="p1")
        _, rec = s.add("Le portail est vert", "fait")
        self.assertEqual(rec["o"], "p1")
        s.update(rec["id"], {"t": "foyer"})
        self.assertEqual(rec["o"], "")

    def test_fusion_de_deux_personnes(self):
        self.i.identifier("Olivier", "Martin")
        self.i.identifier("Olivier", "Dupont")
        self.s.active = "p2"
        self.s.add("Aime le thé", "préférence")
        self.s.active = "p1"
        self.s.add("Aime le café", "préférence")
        self.i.relier("Tiphaine", "femme de", "Olivier Dupont")
        moved = self.s.merge_persons("p2", "p1")
        self.assertEqual(moved, 1)
        self.assertIsNone(self.s.personnes.get("p2"))
        self.assertEqual(len([r for r in self.s.records.values() if r.get("o") == "p1"]), 2)
        self.s.close()
        again = self.make(active="p1")
        self.assertEqual(len(again.search("thé café", touch=False)[0]), 2)

    def test_fusion_invalide(self):
        self.i.identifier("Olivier")
        self.assertEqual(self.s.merge_persons("p1", "p1"), -1)
        self.assertEqual(self.s.merge_persons("p1", "p99"), -1)


if __name__ == "__main__":
    unittest.main()
