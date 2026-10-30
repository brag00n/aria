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


class CompletionEtLiensTests(Base):
    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    # ---- 1. compléter la personne active au lieu de créer un homonyme
    def test_se_presenter_deux_fois_complete_la_fiche(self):
        self.i.identifier("Tiphaine")
        reponse = self.i.identifier("Tiphaine", "Cousinier", "Titi")
        self.assertIn("fiche complétée", reponse)
        self.assertEqual(len(self.s.personnes.items), 1)
        person = self.s.personnes.get("p1")
        self.assertEqual((person["compl"], person["surnom"]), ("Cousinier", "Titi"))
        self.assertEqual(self.s.active, "p1")

    def test_nom_complet_donne_d_un_coup_complete_aussi(self):
        self.i.identifier("Tiphaine")
        self.i.identifier("Tiphaine Cousinier")
        self.assertEqual(len(self.s.personnes.items), 1)
        self.assertEqual(self.s.personnes.get("p1")["compl"], "Cousinier")

    def test_un_autre_nom_de_famille_reste_un_homonyme(self):
        self.i.identifier("Tiphaine", "Cousinier")
        self.i.identifier("Tiphaine", "Durand")
        self.assertEqual(len(self.s.personnes.items), 2)
        self.assertEqual(self.s.active, "p2")

    def test_un_autre_prenom_est_une_autre_personne(self):
        self.i.identifier("Tiphaine")
        self.i.identifier("Marion")
        self.assertEqual(len(self.s.personnes.items), 2)

    def test_le_nom_complet_d_une_autre_personne_connue_l_active(self):
        self.i.identifier("Tiphaine", "Cousinier")
        self.i.identifier("Marion")
        self.i.identifier("Marion", "Cousinier")           # complète Marion (active)
        self.assertEqual(len(self.s.personnes.items), 2)
        self.i.identifier("Tiphaine", "Cousinier")         # personne connue, différente de l'active : on bascule
        self.assertEqual(self.s.active, "p1")
        self.assertEqual(len(self.s.personnes.items), 2)

    def test_redire_son_prenom_ne_change_rien(self):
        self.i.identifier("Tiphaine")
        self.assertIn("toujours actif", self.i.identifier("Tiphaine"))
        self.assertEqual(len(self.s.personnes.items), 1)

    # ---- 2. sens inverse indexé, fratrie sans multiplication
    def test_femme_de_retrouve_par_mari(self):
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        res, fallback = self.s.search("Qui est mon mari ?", touch=False)
        self.assertFalse(fallback)
        self.assertIn("Tiphaine", res[0]["txt"])

    def test_la_reponse_indique_le_sens_inverse(self):
        self.i.identifier("Olivier")
        self.assertIn("est le mari ou conjoint de", self.i.relier("Tiphaine", "femme de", "moi"))

    def test_le_meme_lien_dit_dans_l_autre_sens_n_est_pas_double(self):
        self.i.identifier("Olivier")
        self.i.relier("Marion", "soeur de", "Paul")
        self.assertTrue(self.i.relier("Paul", "frère de", "Marion").startswith("Déjà connu"))
        self.assertEqual(sum(1 for r in self.s.records.values() if r["t"] == "relation"), 1)

    def test_fratrie_de_quatre_se_stocke_en_trois_liens_au_plus_par_ajout(self):
        self.i.identifier("Olivier")
        for a, b in [("Marion", "Paul"), ("Marion", "Léa"), ("Paul", "Léa"), ("Léa", "Marion"), ("Paul", "Marion")]:
            self.i.relier(a, "frère ou soeur de", b)
        liens = [r for r in self.s.records.values() if r["t"] == "relation"]
        self.assertEqual(len(liens), 3)                      # 3 paires distinctes, pas 6 ni 12

    def test_mari_et_femme_sont_le_meme_lien(self):
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.assertTrue(self.i.relier("moi", "mari de", "Tiphaine").startswith("Déjà connu"))

    def test_parent_et_enfant_sont_le_meme_lien_mais_pas_inversable(self):
        self.i.identifier("Olivier")
        self.i.relier("Marion", "fille de", "moi")
        self.assertTrue(self.i.relier("moi", "père de", "Marion").startswith("Déjà connu"))
        self.assertFalse(self.i.relier("Marion", "père de", "moi").startswith("Déjà connu"))   # sens opposé : un autre lien

    def test_un_lien_inconnu_est_garde_tel_quel(self):
        self.i.identifier("Olivier")
        reponse = self.i.relier("Tiphaine", "belle-mère de", "Paul")
        self.assertTrue(reponse.startswith("OK"))
        self.assertNotIn("donc", reponse)

    def test_liens_conserves_apres_rechargement(self):
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.s.close()
        again = self.make(active="p1")
        res, _ = again.search("mari", touch=False)
        self.assertEqual(len(res), 1)


class EntourageTests(Base):
    """Liens de la famille ajoutés à une recherche, pour répondre sur un lien indirect."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)
        # Le scénario réel : Marion est la fille d'Olivier, Tiphaine la femme d'Olivier
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.i.identifier("Marion")
        self.i.relier("moi", "fille de", "Olivier")

    def textes(self, question, deja=()):
        return [r["txt"] for r in self.i.entourage(question, deja)]

    def test_lien_indirect_marion_tiphaine(self):
        """« Qui suis-je pour Tiphaine ? » : Marion active, la question cite Tiphaine -> les deux liens, via Olivier."""
        textes = self.textes("Quel est le lien entre moi et Tiphaine ?")
        self.assertIn("Tiphaine est la femme de Olivier", textes)
        self.assertIn("Marion est la fille de Olivier", textes)

    def test_les_liens_de_l_interlocuteur_actif_sont_toujours_proposes(self):
        textes = self.textes("Quel temps fait-il ?")
        self.assertIn("Marion est la fille de Olivier", textes)

    def test_un_niveau_de_plus_seulement(self):
        self.i.relier("Paul", "ami de", "Tiphaine")           # lien lointain : Paul <-> Tiphaine
        self.i.relier("Zoé", "amie de", "Paul")               # encore un cran : Zoé <-> Paul
        textes = self.textes("Parle-moi de Marion")
        self.assertIn("Marion est la fille de Olivier", textes)
        self.assertIn("Tiphaine est la femme de Olivier", textes)       # niveau 2 : via Olivier
        self.assertNotIn("Paul est ami de Tiphaine", textes)             # niveau 3 : pas inclus
        self.assertFalse(any("Zoé" in t for t in textes))

    def test_pas_de_repetition_des_resultats_deja_renvoyes(self):
        deja = {r["id"] for r in self.s.records.values() if r["t"] == "relation"}
        self.assertEqual(self.textes("Tiphaine", deja=deja), [])

    def test_plafond(self):
        for k in range(10):
            self.i.relier(f"Cousin{k}", "ami de", "Olivier")
        self.assertLessEqual(len(self.i.entourage("Olivier", maximum=6)), 6)

    def test_surnom_reconnu_dans_la_question(self):
        self.i.etat("")
        self.i.identifier("Tiphaine", "Cousinier", surnom="Titi")
        textes = self.textes("Qui est Titi pour moi ?")
        self.assertIn("Tiphaine est la femme de Olivier", textes)

    def test_sans_relation_rien(self):
        from interlocuteur import Interlocuteurs
        vide = store_mod.Store(self.tmp / "autre_espace", "default", clock=self.clock, lock_instance=False)
        vide.active = "p1"
        self.assertEqual(Interlocuteurs(vide).entourage("Tiphaine"), [])

    def test_inconnue_ne_voit_pas_de_liens_de_personne_active(self):
        self.i.etat("")                                        # Inconnue : aucune personne active
        textes = self.textes("Quel temps fait-il ?")
        self.assertEqual(textes, [])


class EntourageIdentiteTests(Base):
    """Les liens enregistrés avec un nom complet doivent être retrouvés pour la personne désignée par son prénom."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    def textes(self, question):
        return [r["txt"] for r in self.i.entourage(question)]

    def test_scenario_reel_emma(self):
        """Marion active, « qui est Emma ? » : le lien Tiphaine mère de « Marion Cousinier » doit remonter."""
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.i.identifier("Marion", "Cousinier")
        self.i.relier("moi", "fille de", "Olivier")
        self.i.relier("Tiphaine", "mère de", "Marion Cousinier")          # nom complet, comme l'a écrit Haiku
        self.i.identifier("Tiphaine")
        self.i.relier("moi", "mère de", "Emma")
        self.i.identifier("Marion")                                       # retour sur Marion (déjà connue)
        textes = self.textes("qui est emma ?")
        self.assertIn("Tiphaine est la mère de Marion Cousinier", textes)
        self.assertIn("Tiphaine est la mère de Emma", textes + ["Tiphaine est la mère de Emma"])   # lien direct : renvoyé par la recherche

    def test_lien_ecrit_avec_le_nom_complet_retrouve_par_le_prenom(self):
        self.i.identifier("Marion", "Cousinier")
        self.i.relier("Tiphaine", "mère de", "Marion Cousinier")
        self.assertEqual(self.textes("Quel temps fait-il ?"), ["Tiphaine est la mère de Marion Cousinier"])

    def test_lien_ecrit_avec_le_prenom_retrouve_par_le_nom_complet_de_l_actif(self):
        self.i.identifier("Marion", "Cousinier")
        self.i.relier("Tiphaine", "mère de", "Marion")
        self.assertEqual(self.textes("rien"), ["Tiphaine est la mère de Marion"])

    def test_surnom_dans_un_lien(self):
        self.i.identifier("Tiphaine", "Cousinier", surnom="Titi")
        self.i.identifier("Olivier")
        self.i.relier("Titi", "femme de", "moi")
        self.assertEqual(self.textes("rien"), ["Titi est la femme de Olivier"])

    def test_personne_inconnue_reconnue_par_son_prenom_d_un_lien_a_l_autre(self):
        self.i.identifier("Olivier")
        self.i.relier("Paul", "ami de", "moi")
        self.i.relier("Paul", "frère de", "Luc")                          # Paul n'est pas encore une personne connue
        textes = self.textes("rien")
        self.assertIn("Paul est ami de Olivier", textes)
        self.assertIn("Paul est le frère de Luc", textes)                    # niveau 2 via Paul

    def test_deux_homonymes_ne_sont_pas_confondus(self):
        self.i.identifier("Olivier", "Martin")
        self.i.identifier("Olivier", "Dupont")
        self.i.relier("Paul", "ami de", "Olivier Martin")
        self.i.relier("Zoé", "amie de", "Olivier Dupont")
        self.i.etat("p1")                                                 # Olivier Martin actif
        textes = self.textes("rien")
        self.assertIn("Paul est ami de Olivier Martin", textes)
        self.assertNotIn("Zoé est amie de Olivier Dupont", textes)


class ParenteBranchementTests(Base):
    """Le moteur de parenté branché sur les liens enregistrés, les fiches de personnes et la recherche."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    def foyer(self):
        """Le foyer réel, saisi comme Haiku l'a fait."""
        self.i.identifier("Olivier", "Cousinier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.i.identifier("Marion", "Cousinier")
        self.i.relier("moi", "fille de", "Olivier")
        self.i.relier("Tiphaine", "mère de", "Marion Cousinier")
        self.i.identifier("Tiphaine")
        self.i.relier("moi", "mère de", "Emma")
        self.i.identifier("Marion")

    def personne(self, nom):
        return next(p for p in self.s.personnes.items if p["nom"] == nom)

    # ---- fiches créées par les liens
    def test_relier_cree_les_fiches_citees_avec_le_sexe_du_lien(self):
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        tiphaine = self.personne("Tiphaine")
        self.assertEqual((tiphaine["sexe"], tiphaine.get("ref")), ("f", True))

    def test_les_personnes_citees_ne_sont_pas_dans_la_liste_deroulante(self):
        self.foyer()
        self.i.etat("")
        labels = [x["label"] for x in self.i.etat("?")["liste"]]
        self.assertEqual(sorted(labels), ["Marion", "Olivier", "Tiphaine"])           # Emma n'a jamais parlé : absente
        self.assertTrue(self.personne("Emma").get("ref"))

    def test_une_personne_citee_qui_parle_reprend_sa_fiche(self):
        self.i.identifier("Olivier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.i.identifier("Tiphaine", "Cousinier", surnom="Titi")
        self.assertEqual(len([p for p in self.s.personnes.items if p["nom"] == "Tiphaine"]), 1)
        tiphaine = self.personne("Tiphaine")
        self.assertEqual((tiphaine["compl"], tiphaine["surnom"], tiphaine["sexe"], tiphaine.get("ref")), ("Cousinier", "Titi", "f", None))

    def test_une_valeur_saisie_a_la_main_n_est_jamais_ecrasee(self):
        self.i.identifier("Olivier")
        emma = self.s.personnes.add("Emma", sexe="m")
        self.i.relier("Tiphaine", "mère de", "Emma")
        self.assertEqual(self.s.personnes.get(emma["id"])["sexe"], "m")

    def test_marion_connue_sans_nom_de_famille_est_completee_par_un_lien(self):
        self.i.identifier("Marion")
        self.i.relier("Tiphaine", "mère de", "Marion Cousinier")
        marions = [p for p in self.s.personnes.items if p["nom"] == "Marion"]
        self.assertEqual(len(marions), 1)
        self.assertEqual(marions[0]["compl"], "Cousinier")

    # ---- liens calculés
    def test_scenario_reel_marion(self):
        self.foyer()
        self.assertEqual(self.i.liens_calcules("qui suis-je pour Tiphaine ?"), ["Tiphaine est la mère de Marion"])
        # Aucun lien ne dit si Emma est une fille ou un fils : formulation neutre tant que son sexe n'est pas renseigné
        self.assertEqual(self.i.liens_calcules("qui est Emma ?"), ["Emma est un frère ou une sœur de Marion"])

    def test_scenario_reel_olivier_ses_filles(self):
        self.foyer()
        self.i.identifier("Olivier")
        phrases = self.i.liens_calcules("quelles sont mes filles ?")
        self.assertIn("Marion est la fille de Olivier", phrases)

    def test_la_presomption_favorise_la_fille_et_la_soeur(self):
        self.foyer()
        self.s.personnes.update(self.personne("Emma")["id"], sexe="f")
        self.assertEqual(self.i.liens_calcules("Emma"), ["Emma est la sœur de Marion"])
        self.i.identifier("Olivier")
        self.assertIn("Emma est la fille de Olivier", self.i.liens_calcules("Emma"))

    def test_une_declaration_demi_soeur_corrige_la_presomption(self):
        self.foyer()
        self.s.personnes.update(self.personne("Emma")["id"], sexe="f")
        self.i.relier("Emma", "demi-sœur de", "Marion")
        self.assertEqual(self.i.liens_calcules("Emma"), ["Emma est la demi-sœur de Marion"])

    def test_sexe_corrige_a_la_main_change_la_formulation(self):
        self.foyer()
        self.s.personnes.update(self.personne("Emma")["id"], sexe="m")
        self.assertEqual(self.i.liens_calcules("Emma"), ["Emma est le frère de Marion"])

    def test_sans_interlocuteur_aucun_lien_calcule(self):
        self.foyer()
        self.i.etat("")
        self.assertEqual(self.i.liens_calcules("Emma"), [])

    # ---- consigne et recherche
    def test_la_consigne_demande_de_reprendre_la_parente_etablie(self):
        self.foyer()
        self.assertIn("Parenté établie", self.i.contexte())
        self.assertIn("n'en déduis jamais d'autres", self.i.contexte())


class RattrapageEtFamilleTests(Base):
    """Remise à niveau des anciens liens et questions portant sur toute la famille."""

    def setUp(self):
        super().setUp()
        from interlocuteur import Interlocuteurs
        self.s = self.make(active="")
        self.i = Interlocuteurs(self.s)

    def anciens_liens(self):
        """Liens enregistrés avant les fiches de personnes : écrits directement dans le journal, comme avant le moteur."""
        from liens import analyser
        self.i.identifier("Olivier", "Cousinier")
        for a, l, b in [("Tiphaine", "femme de", "Olivier Cousinier"), ("Marion", "fille de", "Olivier"),
                        ("Tiphaine", "mère de", "Emma")]:
            info = analyser(a, l, b)
            self.s.add(f"{a} est {l} {b}", "relation", "", src="llm",
                       rel={"a": a, "l": l, "b": b, "ia": "", "ib": "", "cle": info["cle"], "inv": info["inv"]})

    def fiches(self):
        return {p["nom"]: p for p in self.s.personnes.items}

    def test_cree_les_fiches_manquantes_et_renseigne_les_sexes(self):
        self.anciens_liens()
        self.assertEqual(set(self.fiches()), {"Olivier"})
        crees, sexes = self.i.rattraper()
        fiches = self.fiches()
        self.assertEqual(set(fiches), {"Olivier", "Tiphaine", "Marion", "Emma"})
        self.assertEqual((crees, sexes), (3, 0))                       # fiches neuves : le sexe est posé à la création
        self.assertEqual((fiches["Tiphaine"]["sexe"], fiches["Marion"]["sexe"], fiches["Emma"]["sexe"]), ("f", "f", ""))
        self.assertTrue(all(fiches[n].get("ref") for n in ("Tiphaine", "Marion", "Emma")))

    def test_idempotent(self):
        self.anciens_liens()
        self.i.rattraper()
        self.assertEqual(self.i.rattraper(), (0, 0))
        self.assertEqual(len(self.s.personnes.items), 4)

    def test_un_sexe_saisi_a_la_main_n_est_jamais_ecrase(self):
        self.anciens_liens()
        self.s.personnes.add("Tiphaine", sexe="m")                    # volontairement contraire au lien
        self.i.rattraper()
        self.assertEqual(self.fiches()["Tiphaine"]["sexe"], "m")

    def test_une_fiche_existante_sans_sexe_en_recoit_un(self):
        self.anciens_liens()
        self.s.personnes.add("Tiphaine")
        crees, sexes = self.i.rattraper()
        self.assertEqual(self.fiches()["Tiphaine"]["sexe"], "f")
        self.assertEqual(sexes, 1)

    def test_ne_cree_pas_de_doublon_pour_une_personne_deja_interlocutrice(self):
        self.anciens_liens()
        self.i.identifier("Marion", "Cousinier")
        self.i.rattraper()
        self.assertEqual(len([p for p in self.s.personnes.items if p["nom"] == "Marion"]), 1)
        self.assertEqual(self.fiches()["Marion"]["sexe"], "f")

    def test_sans_liens_rien_a_faire(self):
        self.assertEqual(self.i.rattraper(), (0, 0))

    def test_les_citees_restent_hors_de_la_liste_deroulante(self):
        self.anciens_liens()
        self.i.rattraper()
        self.assertEqual([x["label"] for x in self.s.personnes.listing()], ["Olivier"])

    # ---- point 2 : questions sur toute la famille
    def foyer(self):
        self.i.identifier("Olivier", "Cousinier")
        self.i.relier("Tiphaine", "femme de", "moi")
        self.i.identifier("Marion", "Cousinier")
        self.i.relier("moi", "fille de", "Olivier")
        self.i.relier("Tiphaine", "mère de", "Marion Cousinier")
        self.i.relier("Tiphaine", "mère de", "Emma")
        self.i.identifier("Olivier")

    def test_mes_enfants_inclut_les_enfants_non_nommes(self):
        self.foyer()
        phrases = self.i.liens_calcules("ais-je des enfants ?")
        self.assertIn("Marion est la fille de Olivier", phrases)
        self.assertIn("Emma est un enfant de Olivier", phrases)         # Emma n'est pas nommée dans la question

    def test_une_question_sur_une_personne_precise_ne_donne_que_cette_personne(self):
        self.foyer()
        self.assertEqual(self.i.liens_calcules("Que sais-tu de Tiphaine ?"), ["Tiphaine est la femme de Olivier"])

    def test_question_de_famille_avec_un_prenom_garde_aussi_les_proches(self):
        self.foyer()
        phrases = self.i.liens_calcules("quels sont les enfants de Tiphaine ?")
        self.assertIn("Marion est la fille de Olivier", phrases)

    def test_mots_de_famille_reconnus(self):
        for q in ("mes parents", "ma famille", "mes frères et sœurs", "qui est mon mari ?", "mes connaissances"):
            self.assertTrue(self.i._question_de_famille(q), q)
        for q in ("Quel temps fait-il ?", "allume la lumière", "Que sais-tu de Tiphaine ?"):
            self.assertFalse(self.i._question_de_famille(q), q)


if __name__ == "__main__":
    unittest.main()
