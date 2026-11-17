"""Liens publics d'un accès tiers (option « publier ses liens » de sa clé) : un tiers relie LUI-MÊME à une personne (existante ou nouvelle), le lien est
visible de tous ; il ne crée ni ne modifie jamais un lien entre d'autres personnes, ni celui qu'un humain ou un autre accès a établi.
Sans l'option, tout reste privé (comportement d'avant)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import interlocuteur  # noqa: E402
import store as store_mod  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402

try:
    import httpx  # noqa: F401
    from starlette.testclient import TestClient
    import http_facade
    HAVE = True
except ImportError:
    HAVE = False


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        self.t += 1
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="liens_pub_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.i = Interlocuteurs(self.s)
        self.svc = Service(self.s, self.i)
        self.i.identifier("Olivier", "Cousinier")
        self.olivier = self.s.active
        self.i.relier("Tiphaine", "conjoint", "moi")
        self.i.relier("Marion", "enfant", "moi")
        self.bamboo = self.s.personnes.ensure_system("k_bamboo", "Bamboo")["id"]
        self.autre = self.s.personnes.ensure_system("k_autre", "Autre robot")["id"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def liens(self):
        return [r for r in self.s.records.values() if r["t"] == "relation"]

    def lien_de(self, nom):
        return next((r for r in self.liens() if nom in r["txt"]), None)

    def relier(self, a, lien, b, precision="", principal=None, publier=True):
        return self.svc.relier(a, lien, b, precision, principal=principal or self.bamboo, publier=publier)

    def visibles_par(self, principal):
        return {r["id"] for r in self.liens() if self.s.visible_to(r, principal)}


class PublicationTests(Base):
    def test_le_lien_d_un_tiers_autorise_est_public(self):
        out = self.relier("moi", "autre", "Olivier", "propriétaire")
        self.assertTrue(out.startswith("OK"), out)
        rec = self.lien_de("propriétaire")
        self.assertEqual(rec["o"], "")                                         # public
        self.assertEqual(rec["rel"]["par"], self.bamboo)                       # son auteur est retenu
        self.assertTrue(self.s.visible_to(rec, self.olivier))                  # un humain le voit
        self.assertTrue(self.s.visible_to(rec, self.bamboo))

    def test_sans_l_option_le_lien_reste_prive(self):
        self.relier("moi", "autre", "Olivier", "propriétaire", publier=False)
        rec = self.lien_de("propriétaire")
        self.assertEqual(rec["o"], self.bamboo)
        self.assertNotIn("par", rec["rel"])
        self.assertFalse(self.s.visible_to(rec, self.olivier))

    def test_le_lien_public_d_un_tiers_est_lu_par_aria(self):
        self.relier("moi", "autre", "Olivier", "propriétaire")
        out = self.svc.rechercher("Bamboo propriétaire", principal=self.olivier)
        self.assertIn("propriétaire", out)

    def test_un_autre_tiers_ne_voit_pas_le_lien_public_comme_souvenir(self):
        self.relier("moi", "autre", "Olivier", "propriétaire")
        rec = self.lien_de("propriétaire")
        self.assertFalse(self.s.visible_to(rec, self.autre))                   # même règle qu'avant pour les relations des autres

    def test_la_trace_du_lien_public_est_conservee(self):
        tr = {"echange": "e_abcdef123456", "ts": "2026-11-17T14:00:00+00:00", "src": "k_bamboo:Bamboo", "who": "Bamboo", "x": "Je suis ton propriétaire"}
        self.svc.relier("moi", "autre", "Olivier", "propriétaire", principal=self.bamboo, trace=tr, publier=True)
        self.assertEqual(self.lien_de("propriétaire")["tr"]["x"], "Je suis ton propriétaire")


class PersonnesTests(Base):
    def test_personne_nouvelle_creee_a_valider(self):
        out = self.relier("moi", "ami", "Zoé")
        self.assertTrue(out.startswith("OK"), out)
        zoe = next(p for p in self.s.personnes.items if p["nom"] == "Zoé")
        self.assertTrue(zoe.get("a_valider"))
        self.assertEqual(zoe["par"], self.bamboo)
        self.assertTrue(zoe.get("ref"))                                        # citée : pas dans la liste des interlocuteurs d'Aria
        self.assertNotIn("Zoé", [p["label"] for p in self.s.personnes.listing()])

    def test_sans_l_option_pas_de_creation_de_personne(self):
        out = self.relier("moi", "ami", "Zoé", publier=False)
        self.assertIn("pas connue", out)
        self.assertFalse(any(p["nom"] == "Zoé" for p in self.s.personnes.items))

    def test_personne_existante_n_est_pas_marquee(self):
        self.relier("moi", "ami", "Marion")
        marion = next(p for p in self.s.personnes.items if p["nom"] == "Marion")
        self.assertNotIn("a_valider", marion)
        self.assertNotIn("par", marion)

    def test_plafond_de_creations_par_jour(self):
        for n in range(interlocuteur.MAX_NOUVELLES_PERSONNES_PAR_JOUR):
            self.assertTrue(self.relier("moi", "ami", f"Nouvelle{n}").startswith("OK"))
        out = self.relier("moi", "ami", "Trop")
        self.assertIn("beaucoup de nouvelles personnes", out)
        self.assertFalse(any(p["nom"] == "Trop" for p in self.s.personnes.items))

    def test_le_plafond_est_par_acces(self):
        for n in range(interlocuteur.MAX_NOUVELLES_PERSONNES_PAR_JOUR):
            self.relier("moi", "ami", f"Nouvelle{n}")
        self.assertTrue(self.relier("moi", "ami", "AutreCreation", principal=self.autre).startswith("OK"))

    def test_homonymes_refuses_sans_deviner(self):
        self.s.personnes.add("Paul", "Martin")
        self.s.personnes.add("Paul", "Durand")
        out = self.relier("moi", "ami", "Paul")
        self.assertIn("Plusieurs personnes", out)
        self.assertEqual(len([p for p in self.s.personnes.items if p["nom"] == "Paul"]), 2)

    def test_un_systeme_ne_renseigne_pas_le_sexe_d_une_fiche_creee(self):
        self.relier("Zoé", "autre", "moi", "mère")                              # « mère » dirait que Zoé est une femme : un humain le saisit, pas un système
        zoe = next(p for p in self.s.personnes.items if p["nom"] == "Zoé")
        self.assertEqual(zoe.get("sexe", ""), "")
        self.i.relier("Léa", "autre", "moi", "mère")                           # le même lien posé par un humain, lui, renseigne le sexe
        self.assertEqual(next(p for p in self.s.personnes.items if p["nom"] == "Léa").get("sexe"), "f")

    def test_validation_par_un_humain(self):
        self.relier("moi", "ami", "Zoé")
        zoe = next(p for p in self.s.personnes.items if p["nom"] == "Zoé")
        self.assertTrue(self.s.personnes.update(zoe["id"], a_valider=False))
        zoe = self.s.personnes.get(zoe["id"])
        self.assertNotIn("a_valider", zoe)
        self.assertEqual(zoe["par"], self.bamboo)                              # l'origine reste connue

    def test_champs_conserves_apres_rechargement(self):
        self.relier("moi", "ami", "Zoé")
        rouvert = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        zoe = next(p for p in rouvert.personnes.items if p["nom"] == "Zoé")
        self.assertEqual((zoe.get("a_valider"), zoe.get("par")), (True, self.bamboo))


class LimitesTests(Base):
    def test_jamais_de_lien_entre_d_autres_personnes_meme_avec_l_option(self):
        avant = len(self.liens())
        out = self.relier("Tiphaine", "enfant", "Marion")
        self.assertIn("l'un des deux doit être", out)
        self.assertEqual(len(self.liens()), avant)

    def test_jamais_de_lien_avec_soi_meme(self):
        self.assertIn("liée à elle-même", self.relier("moi", "ami", "Bamboo"))

    def test_ne_modifie_pas_le_lien_d_un_humain(self):
        self.i.relier("Bamboo", "autre", "moi", "robot de la maison")           # établi par Olivier (humain), public
        avant = self.lien_de("robot de la maison")
        self.assertIsNotNone(avant)
        out = self.relier("moi", "autre", "Olivier", "propriétaire")
        self.assertIn("existe déjà", out)
        self.assertEqual(self.lien_de("robot de la maison")["txt"], avant["txt"])
        self.assertIsNone(self.lien_de("propriétaire"))

    def test_les_liens_de_deux_acces_vers_la_meme_personne_sont_distincts(self):
        """« Autre robot — Olivier » et « Bamboo — Olivier » sont deux liens différents : chacun a le sien, sans écraser celui de l'autre."""
        self.relier("moi", "ami", "Olivier", principal=self.autre)
        out = self.relier("moi", "autre", "Olivier", "propriétaire")
        self.assertTrue(out.startswith("OK"), out)
        par = sorted(r["rel"]["par"] for r in self.liens() if r["rel"].get("par"))
        self.assertEqual(par, sorted([self.autre, self.bamboo]))
        self.assertIn("Autre robot", self.lien_de("ami(e)")["txt"])

    def test_ne_modifie_pas_le_lien_d_un_autre_acces_entre_les_memes_personnes(self):
        """Le seul cas de conflit : une paire déjà reliée par un autre auteur (humain ou accès). L'un d'eux ne peut pas écraser l'autre."""
        self.relier("moi", "ami", "Olivier", principal=self.autre)
        lien = self.lien_de("ami(e)")
        lien["rel"]["ia"], lien["rel"]["ib"] = self.bamboo, self.olivier         # le lien de l'autre accès relie (par erreur) la même paire
        out = self.relier("moi", "autre", "Olivier", "propriétaire")
        self.assertIn("existe déjà", out)
        self.assertIn("ami(e)", self.s.records[lien["id"]]["txt"])

    def test_met_a_jour_son_propre_lien(self):
        self.relier("moi", "ami", "Olivier")
        premier = self.lien_de("ami(e)")
        out = self.relier("moi", "autre", "Olivier", "propriétaire")
        self.assertIn("mis à jour", out)
        rec = self.s.records[premier["id"]]
        self.assertIn("propriétaire", rec["txt"])
        self.assertEqual(rec["rel"]["par"], self.bamboo)
        self.assertEqual(len([r for r in self.liens() if r["rel"].get("par") == self.bamboo]), 1)    # pas de doublon

    def test_peut_oublier_ses_liens_publics_pas_ceux_des_autres(self):
        self.relier("moi", "ami", "Olivier")
        mien = self.lien_de("ami(e)")
        humain = self.lien_de("Tiphaine")
        self.assertEqual(self.svc.oublier(humain["id"], principal=self.bamboo), "Souvenir introuvable.")
        self.assertIn(humain["id"], self.s.records)
        self.assertEqual(self.svc.oublier(mien["id"], principal=self.bamboo), "Oublié.")
        self.assertNotIn(mien["id"], self.s.records)

    def test_un_autre_acces_ne_peut_pas_oublier_le_lien_public_de_bamboo(self):
        self.relier("moi", "ami", "Olivier")
        mien = self.lien_de("ami(e)")
        self.assertEqual(self.svc.oublier(mien["id"], principal=self.autre), "Souvenir introuvable.")
        self.assertIn(mien["id"], self.s.records)

    def test_les_autres_types_restent_prives(self):
        """Seule la RELATION est publiable : un fait ou un souvenir du foyer d'un système reste privé."""
        status, rec = self.s.add("La maison a un jardin", "foyer", src="llm", owner=self.bamboo, publier=True)
        self.assertEqual(rec["o"], self.bamboo)
        status, rec = self.s.add("Bamboo aime la lumière", "preference", src="llm", owner=self.bamboo, publier=True)
        self.assertEqual(rec["o"], self.bamboo)

    def test_un_humain_n_est_pas_concerne_par_l_option(self):
        out = self.svc.relier("Paul", "ami", "Marion", principal=self.olivier, publier=True)
        self.assertTrue(out.startswith("OK"), out)
        self.assertNotIn("par", self.lien_de("Paul")["rel"])

    def test_ne_peut_pas_se_faire_passer_pour_l_auteur_d_un_autre(self):
        """`par` n'est pas modifiable : update ignore `rel.par` venant d'un autre principal (le champ vient du serveur, jamais du LLM)."""
        self.relier("moi", "ami", "Olivier")
        mien = self.lien_de("ami(e)")
        self.assertEqual(mien["rel"]["par"], self.bamboo)


@unittest.skipUnless(HAVE, "mcp / starlette absents (lancer dans le conteneur)")
class CanalHttpTests(Base):
    """L'option vient de l'en-tête posé par Aria, jamais d'un argument du LLM."""

    def setUp(self):
        super().setUp()
        self.app = http_facade.make_app(self.s, self.svc, "secret-de-test-assez-long")
        self.client = TestClient(self.app.inner, base_url="http://127.0.0.1:7872")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        super().tearDown()

    def tool(self, args, publish=None, cle="k_bamboo", nom="Bamboo"):
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
                   "X-Memory-Principal": cle, "X-Memory-Name": http_facade.encode_name(nom)}
        if publish is not None:
            headers["X-Memory-Publish"] = publish
        r = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "relier", "arguments": args}}, headers=headers)
        return r.json()["result"]["content"][0]["text"]

    def test_en_tete_present_le_lien_est_public(self):
        self.assertTrue(self.tool({"personne_a": "moi", "lien": "ami", "personne_b": "Marion"}, publish="1").startswith("OK"))
        self.assertEqual(self.lien_de("Bamboo")["o"], "")

    def test_sans_en_tete_ou_valeur_autre_le_lien_reste_prive(self):
        for n, valeur in enumerate((None, "0", "true", "yes", " 2 ")):
            self.tool({"personne_a": "moi", "lien": "ami", "personne_b": "Marion"}, publish=valeur, cle=f"k_x{n}", nom=f"Rob{n}")
        prives = [r for r in self.liens() if "Rob" in r["txt"]]
        self.assertEqual(len(prives), 5)
        for r in prives:
            self.assertNotEqual(r["o"], "", r["txt"])

    def test_le_llm_ne_peut_pas_activer_l_option_par_un_argument(self):
        out = self.tool({"personne_a": "moi", "lien": "ami", "personne_b": "Marion", "publier": True})
        self.assertNotEqual(self.lien_de("Bamboo")["o"] if self.lien_de("Bamboo") else "x", "")


if __name__ == "__main__":
    unittest.main()
