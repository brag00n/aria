"""Principaux : humains et personnes SYSTÈME (agent). Un système connaît toutes les personnes et leurs liens, mais ne lit que ses propres
souvenirs (et le foyer partagé) ; il n'écrit jamais dans le partagé, ne peut ni changer d'identité ni relier deux humains."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store as store_mod  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        self.t += 1
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="principaux_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.i = Interlocuteurs(self.s)
        self.svc = Service(self.s, self.i)
        # foyer : Olivier (humain, actif d'Aria), Tiphaine et Marion ; liens ; souvenirs privés et partagés
        self.i.identifier("Olivier", "Cousinier")
        self.olivier = self.s.active
        self.i.relier("Tiphaine", "conjoint", "moi")
        self.i.relier("Marion", "enfant", "moi")
        self.s.add("Olivier aime les films de science-fiction", "preference")           # privé d'Olivier
        self.s.add("La chatte s'appelle Réglisse", "foyer")                              # partagé, type foyer
        self.s.add("Olivier prend rendez-vous chez le dentiste", "tache")               # privé d'Olivier
        self.s.add("Ancien souvenir partagé du foyer sans type foyer", "fait", owner=None)  # sera rendu partagé ci-dessous
        ancien = next(r for r in self.s.records.values() if r["txt"].startswith("Ancien"))
        ancien["o"] = ""                                                                 # comme les souvenirs d'avant les propriétaires
        self.robot = self.s.personnes.ensure_system("k_robot1", "Robot")["id"]
        self.cam = self.s.personnes.ensure_system("k_camera", "Caméra")["id"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def textes(self, resultat):
        return resultat


class PersonneSystemeTests(Base):
    def test_identite_liee_a_la_cle_interne_pas_au_nom(self):
        avant = self.s.personnes.systeme("k_robot1")["id"]
        renomme = self.s.personnes.ensure_system("k_robot1", "Bamboo 4WD")
        self.assertEqual(renomme["id"], avant)                                            # même fiche, même id : les souvenirs suivent
        self.assertEqual(renomme["nom"], "Bamboo 4WD")
        self.assertEqual(len([p for p in self.s.personnes.items if p.get("cle") == "k_robot1"]), 1)

    def test_deux_acces_deux_personnes(self):
        self.assertNotEqual(self.robot, self.cam)
        self.assertEqual(self.s.personnes.systeme("k_camera")["nom"], "Caméra")

    def test_le_systeme_n_est_pas_un_interlocuteur_d_aria(self):
        ids = {p["id"] for p in self.s.personnes.listing()}
        self.assertNotIn(self.robot, ids)
        self.assertNotIn(self.robot, {p["id"] for p in self.s.personnes.listing(avec_citees=True)})
        self.assertNotIn(self.robot, {p["id"] for p in self.i.etat("?")["liste"]})
        self.assertIn(self.olivier, ids)

    def test_retrocompatibilite_un_fichier_sans_type_est_humain(self):
        brut = (self.tmp / "default" / "personnes.jsonl").read_text("utf-8")
        lignes = [json.loads(l) for l in brut.splitlines()]
        humains = [p for p in lignes if p.get("type") != "systeme"]
        self.assertTrue(humains and all("type" not in p and "cle" not in p for p in humains))        # aucun champ ajouté aux humains
        s2 = store_mod.Store(self.tmp, "default", lock_instance=False)
        self.assertEqual(s2.personnes.get(self.olivier)["nom"], "Olivier")
        self.assertEqual(s2.personnes.get(self.robot).get("type"), "systeme")                 # et le type système survit au rechargement

    def test_cle_ou_nom_vide_refuses(self):
        self.assertIsNone(self.s.personnes.ensure_system("", "X"))
        self.assertIsNone(self.s.personnes.ensure_system("k_x", "  "))


class SouvenirsTests(Base):
    def test_un_systeme_ne_lit_pas_les_souvenirs_prives_d_un_humain(self):
        out = self.svc.rechercher("films science-fiction dentiste rendez-vous", principal=self.robot)
        for prive in ("science-fiction", "dentiste"):
            self.assertNotIn(prive, out)

    def test_un_humain_ne_lit_pas_les_souvenirs_d_un_systeme(self):
        self.svc.memoriser("La batterie du robot est à 80 pour cent", "fait", principal=self.robot)
        self.assertNotIn("batterie", self.svc.rechercher("batterie du robot", principal=self.olivier))
        self.s.active = self.olivier
        self.assertNotIn("batterie", self.svc.rechercher("batterie du robot"))              # même en stdio, comme Aria

    def test_deux_systemes_ne_se_lisent_pas(self):
        self.svc.memoriser("Le robot a vu un chat dans le jardin", principal=self.robot)
        self.svc.memoriser("La caméra du garage est tournée vers la porte", principal=self.cam)
        self.assertIn("chat dans le jardin", self.svc.rechercher("jardin chat", principal=self.robot))
        self.assertNotIn("garage", self.svc.rechercher("caméra garage porte", principal=self.robot))
        self.assertNotIn("jardin", self.svc.rechercher("robot jardin chat", principal=self.cam))

    def test_un_systeme_lit_le_foyer_partage(self):
        self.assertIn("Réglisse", self.svc.rechercher("chatte Réglisse", principal=self.robot))

    def test_un_systeme_ne_lit_pas_les_anciens_souvenirs_partages_hors_foyer(self):
        self.assertNotIn("Ancien souvenir", self.svc.rechercher("ancien souvenir partagé du foyer", principal=self.robot))
        self.s.active = self.olivier
        self.assertIn("Ancien souvenir", self.svc.rechercher("ancien souvenir partagé du foyer"))          # un humain, lui, le voit toujours

    def test_toutes_les_ecritures_d_un_systeme_sont_privees_meme_foyer_et_relation(self):
        for type_ in ("fait", "preference", "tache", "foyer"):
            self.svc.memoriser(f"Note {type_} du robot", type_, principal=self.robot)
        mes = [r for r in self.s.records.values() if "du robot" in r["txt"]]
        self.assertEqual(len(mes), 4)
        self.assertTrue(all(r["o"] == self.robot for r in mes), [(r["t"], r["o"]) for r in mes])
        self.s.active = self.olivier
        self.assertNotIn("du robot", self.svc.rechercher("Note foyer du robot"))                 # le foyer partagé n'est pas pollué

    def test_un_systeme_n_oublie_que_ses_souvenirs(self):
        mien = self.svc.memoriser("Mon souvenir à moi", principal=self.robot).split()[1]
        self.assertEqual(self.svc.oublier(mien, principal=self.robot), "Oublié.")
        pref = next(r for r in self.s.records.values() if "science-fiction" in r["txt"])["id"]
        self.assertEqual(self.svc.oublier(pref, principal=self.robot), "Souvenir introuvable.")           # privé d'Olivier
        foyer = next(r for r in self.s.records.values() if "Réglisse" in r["txt"])["id"]
        self.assertEqual(self.svc.oublier(foyer, principal=self.robot), "Souvenir introuvable.")          # partagé : pas le sien
        self.assertIn(pref, self.s.records)
        self.assertIn(foyer, self.s.records)

    def test_doublons_par_principal(self):
        a = self.svc.memoriser("Il pleut", principal=self.robot)
        b = self.svc.memoriser("Il pleut", principal=self.cam)
        self.assertTrue(a.startswith("OK") and b.startswith("OK"))                                       # deux souvenirs, pas un « déjà connu »

    def test_debit_borne_par_principal(self):
        figee = self.s.clock()
        self.s.clock = lambda: figee                                                                      # le temps ne s'écoule pas pendant la rafale
        for n in range(store_mod.WRITES_PER_MINUTE):
            self.svc.memoriser(f"Mesure numéro {n} du robot", principal=self.robot)
        bloque = self.svc.memoriser("Mesure de trop du robot", principal=self.robot)
        self.assertIn("minute", bloque)
        self.assertTrue(self.svc.memoriser("Mesure de trop du robot", principal=self.robot).count("minute"))         # et reste bloqué
        self.assertTrue(self.svc.memoriser("La caméra marche", principal=self.cam).startswith("OK"))      # un autre principal n'est pas bloqué
        self.s.active = self.olivier
        self.assertTrue(self.svc.memoriser("Olivier aime le café").startswith("OK"))                     # ni Aria


class GrapheCommunTests(Base):
    def test_un_systeme_connait_toutes_les_personnes_et_leurs_liens(self):
        out = self.svc.rechercher("Tiphaine Marion famille", principal=self.robot)
        self.assertIn("Tiphaine", out)
        self.assertIn("Marion", out)
        self.assertTrue("Liens connus dans la famille" in out or "Parenté établie" in out)

    def test_le_graphe_de_la_famille_est_identique_pour_un_humain_et_un_systeme(self):
        g_humain = self.i.graphe(self.olivier)
        self.assertEqual({v["nom"] for v in g_humain["voisins"]}, {"Tiphaine", "Marion"})
        relations = [r["txt"] for r in self.s.records.values() if r["t"] == "relation"]
        out = self.svc.rechercher("Tiphaine", principal=self.robot)
        self.assertTrue(any(t in out for t in relations), out)

    def test_la_parente_d_un_systeme_part_de_lui_pas_d_un_humain(self):
        self.i.relier("moi", "ami", "Marion", principal=self.robot)
        out = self.svc.rechercher("Marion", principal=self.robot)
        self.assertIn("Robot", out)


class RelationsTests(Base):
    def test_un_systeme_relie_seulement_lui_meme_a_une_personne_connue(self):
        out = self.svc.relier("moi", "ami", "Marion", principal=self.robot)
        self.assertTrue(out.startswith("OK"), out)
        rec = next(r for r in self.s.records.values() if r["t"] == "relation" and "Robot" in r["txt"])
        self.assertEqual(rec["o"], self.robot)                                                           # sa relation lui appartient
        self.assertEqual({rec["rel"]["ia"], rec["rel"]["ib"]} & {self.robot}, {self.robot})

    def test_un_systeme_ne_cree_pas_de_lien_entre_deux_humains(self):
        avant = len([r for r in self.s.records.values() if r["t"] == "relation"])
        out = self.svc.relier("Tiphaine", "enfant", "Marion", principal=self.robot)
        self.assertIn("l'un des deux doit être", out)
        self.assertEqual(len([r for r in self.s.records.values() if r["t"] == "relation"]), avant)

    def test_un_systeme_ne_cree_pas_de_nouvelle_personne_humaine(self):
        avant = len(self.s.personnes.items)
        out = self.svc.relier("moi", "ami", "Inconnu Total", principal=self.robot)
        self.assertIn("pas connue", out)
        self.assertEqual(len(self.s.personnes.items), avant)

    def test_la_relation_d_un_systeme_n_est_pas_dans_le_foyer_des_humains(self):
        self.svc.relier("moi", "ami", "Marion", principal=self.robot)
        rec = next(r for r in self.s.records.values() if r["t"] == "relation" and "Robot" in r["txt"])
        self.assertTrue(rec["o"])                                                                         # propriétaire = le système (jamais « partagé »)

    def test_un_humain_relie_comme_avant(self):
        self.s.active = self.olivier
        self.assertTrue(self.svc.relier("Paul", "ami", "moi").startswith("OK"))


class UsurpationTests(Base):
    def test_un_systeme_ne_peut_pas_changer_d_identite(self):
        avant = self.s.active
        for nom in ("Olivier", "Tiphaine", "Marion", "Robot", "Nouveau"):
            out = self.svc.identifier(nom, principal=self.robot)
            self.assertIn("identité est fixée", out, nom)
        self.assertEqual(self.s.active, avant)                                                           # l'actif d'Aria n'a pas bougé
        self.assertEqual(len(self.s.personnes.items), len({p["id"] for p in self.s.personnes.items}))
        self.assertNotIn("Nouveau", [p["nom"] for p in self.s.personnes.items])                         # et rien n'a été créé

    def test_un_appel_de_systeme_ne_modifie_jamais_l_actif_d_aria(self):
        self.s.active = self.olivier
        self.svc.memoriser("Mesure", principal=self.robot)
        self.svc.rechercher("Mesure", principal=self.robot)
        self.svc.relier("moi", "ami", "Marion", principal=self.robot)
        self.svc.oublier("m_xxxxxx", principal=self.robot)
        self.assertEqual(self.s.active, self.olivier)

    def test_memoriser_avec_un_principal_inconnu_ou_vide_ne_donne_pas_acces_aux_humains(self):
        for faux in ("p999", "k_robot1", "Olivier", " "):
            out = self.svc.rechercher("films science-fiction", principal=faux)
            self.assertNotIn("science-fiction", out, faux)

    def test_consigne_du_tour_pour_un_systeme(self):
        c = self.i.contexte(self.robot)
        self.assertIn("système", c)
        self.assertIn("Robot", c)
        self.assertNotIn("identifier_interlocuteur", c)
        self.assertIn("Olivier", self.i.contexte(self.olivier))                                          # humain inchangé


class SimultaneiteTests(Base):
    def test_aria_et_deux_systemes_en_meme_temps_sans_se_melanger(self):
        import threading
        self.s.active = self.olivier
        erreurs = []

        def tiers(pid, n):
            try:
                for k in range(8):
                    self.svc.memoriser(f"Observation {n} numéro {k}", principal=pid)
                    self.svc.rechercher(f"Observation {n}", principal=pid)
            except Exception as e:                                                                       # pragma: no cover
                erreurs.append(e)

        ts = [threading.Thread(target=tiers, args=(self.robot, "robot")), threading.Thread(target=tiers, args=(self.cam, "camera"))]
        [t.start() for t in ts]
        for k in range(8):
            self.svc.memoriser(f"Olivier mesure {k}")
        [t.join() for t in ts]
        self.assertEqual(erreurs, [])
        self.assertEqual(self.s.active, self.olivier)
        owners = {}
        for r in self.s.records.values():
            if "Observation robot" in r["txt"]:
                owners.setdefault("robot", set()).add(r["o"])
            if "Observation camera" in r["txt"]:
                owners.setdefault("camera", set()).add(r["o"])
        self.assertEqual(owners, {"robot": {self.robot}, "camera": {self.cam}})


class PageDeGestionTests(Base):
    def test_la_page_de_gestion_voit_tout_avec_le_proprietaire(self):
        self.svc.memoriser("Secret du robot", principal=self.robot)
        tout, _ = self.s.search("Secret du robot", 5, touch=False, everything=True)
        self.assertEqual([r["o"] for r in tout if "Secret du robot" in r["txt"]], [self.robot])


if __name__ == "__main__":
    unittest.main()
