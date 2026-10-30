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


if __name__ == "__main__":
    unittest.main()
