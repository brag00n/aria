"""Droits des agents et agent par défaut « aria » (components/agents/rights.py, keys.py) : modèle de données, verrous, rétrocompatibilité.
Bibliothèque standard seulement."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.agents import keys, rights  # noqa: E402


class DroitsTests(unittest.TestCase):
    def test_un_agent_ordinaire_n_a_aucun_droit_par_defaut(self):
        eff = rights.effective({"id": "k_1"})
        self.assertEqual(set(eff), set(rights.ALL))
        self.assertFalse(any(eff.values()))

    def test_l_agent_systeme_a_tous_les_droits_quoi_que_dise_le_fichier(self):
        for rec in ({"id": "aria", "system": True}, {"id": "aria", "system": True, "rights": {n: False for n in rights.RIGHTS}, "publish_links": False}):
            self.assertTrue(all(rights.effective(rec).values()), rec)
            for n in rights.ALL:
                self.assertTrue(rights.has(rec, n), n)

    def test_droits_accordes_un_par_un(self):
        rec = {"id": "k_1", "rights": {"local_llm": True, "all_mcp": False}, "publish_links": True}
        eff = rights.effective(rec)
        self.assertEqual((eff["local_llm"], eff["all_mcp"], eff["no_limits"], eff["publish_links"], eff["publish_speaker"]), (True, False, False, True, False))

    def test_seul_exactement_true_accorde_un_droit(self):
        for valeur in ("true", "oui", 1, 2, [1], {"a": 1}, "False", None, 0.5):
            self.assertFalse(rights.has({"id": "k_1", "rights": {"local_llm": valeur}}, "local_llm"), repr(valeur))
        self.assertTrue(rights.has({"id": "k_1", "rights": {"local_llm": True}}, "local_llm"))

    def test_nom_inconnu_jamais_accorde_meme_pour_le_systeme(self):
        for nom in ("", "admin", "root", "system", None, "local_llm ", "LOCAL_LLM"):
            self.assertFalse(rights.has({"id": "aria", "system": True}, nom), repr(nom))
            self.assertFalse(rights.has({"id": "k_1", "rights": {"admin": True}}, nom), repr(nom))

    def test_clean_ignore_les_noms_inconnus_et_les_formes_absurdes(self):
        self.assertEqual(rights.clean({"local_llm": True, "admin": True, "all_mcp": "oui"}), {"local_llm": True, "all_mcp": False, "global_speaker": False, "no_limits": False})
        for absurde in (None, [], "x", 3, True):
            self.assertEqual(rights.clean(absurde), {n: False for n in rights.RIGHTS})

    def test_sans_agent(self):
        self.assertFalse(any(rights.effective(None).values()))
        self.assertFalse(rights.has(None, "local_llm"))
        self.assertFalse(rights.is_system(None))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rights_"))
        self.store = keys.KeyStore(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fichier(self):
        return json.loads((self.tmp / "agents.json").read_text("utf-8"))["agents"]


class AgentParDefautTests(Base):
    def test_l_agent_aria_existe_des_le_premier_demarrage(self):
        vue = self.store.get("aria")
        self.assertIsNotNone(vue)
        self.assertEqual((vue["label"], vue["system"], vue["profile"]), ("Aria", True, "aria"))
        self.assertTrue(all(vue["rights"].values()))

    def test_sa_cle_api_est_generee_cote_serveur_et_fonctionne(self):
        cle = self.store.system_key
        self.assertTrue(cle.startswith("ak_"))
        statut, rec = self.store.authenticate(cle)
        self.assertEqual((statut, rec["id"]), ("ok", "aria"))
        self.assertNotIn(cle, (self.tmp / "agents.json").read_text("utf-8"))                      # jamais stockée en clair

    def test_la_cle_en_clair_n_est_connue_que_du_demarrage_qui_l_a_creee(self):
        rouvert = keys.KeyStore(self.tmp)
        self.assertEqual(rouvert.system_key, "")
        self.assertEqual(rouvert.authenticate(self.store.system_key)[0], "ok")                   # mais la clé reste valable

    def test_creation_idempotente(self):
        premiere = self.store.system_key
        self.assertFalse(self.store.ensure_system_agent())
        rouvert = keys.KeyStore(self.tmp)
        self.assertEqual(len([k for k in rouvert.list() if k["id"] == "aria"]), 1)
        self.assertEqual(rouvert.authenticate(premiere)[0], "ok")

    def test_sans_quota_en_pratique(self):
        self.assertGreaterEqual(self.store.get("aria")["limits"]["rate_per_min"], 100_000)

    def test_ne_se_supprime_pas(self):
        self.assertFalse(self.store.delete("aria"))
        self.assertIsNotNone(self.store.get("aria"))
        self.assertIn("aria", [a["id"] for a in self.fichier()])

    def test_ne_se_revoque_pas_mais_se_reactive(self):
        self.assertFalse(self.store.revoke("aria"))
        self.assertFalse(self.store.get("aria")["revoked"])
        self.assertTrue(self.store.revoke("aria", False))                                       # lever une révocation reste permis

    def test_regeneration_autorisee_et_ancienne_cle_invalidee(self):
        ancienne = self.store.system_key
        vue, nouvelle = self.store.regenerate("aria")
        self.assertEqual(vue["id"], "aria")
        self.assertEqual(self.store.authenticate(ancienne)[0], "unknown")
        self.assertEqual(self.store.authenticate(nouvelle)[0], "ok")

    def test_les_champs_verrouilles_ne_changent_pas(self):
        avant = self.store.get("aria")
        apres = self.store.update("aria", rights={n: False for n in rights.RIGHTS}, publish_links=False, publish_speaker=False, profile="autre",
                                  expires_days=1, revoked=True, system=False, id="autre")
        self.assertTrue(all(apres["rights"].values()))
        self.assertEqual((apres["profile"], apres["expires"], apres["revoked"], apres["system"], apres["id"]),
                         (avant["profile"], avant["expires"], False, True, "aria"))
        self.assertTrue(apres["publish_links"] and apres["publish_speaker"])

    def test_les_champs_libres_changent(self):
        vue = self.store.update("aria", label="Aria maison", origins=["https://a.example.com"], voice=False, retention_days=30)
        self.assertEqual((vue["label"], vue["origins"], vue["voice"], vue["retention_days"]), ("Aria maison", ["https://a.example.com"], False, 30))

    def test_le_fichier_edite_a_la_main_ne_retire_pas_le_statut_systeme(self):
        brut = json.loads((self.tmp / "agents.json").read_text("utf-8"))
        for a in brut["agents"]:
            if a["id"] == "aria":
                a.update(system=False, rights={n: False for n in rights.RIGHTS}, publish_links=False)
        (self.tmp / "agents.json").write_text(json.dumps(brut), "utf-8")
        vue = keys.KeyStore(self.tmp).get("aria")
        self.assertTrue(vue["system"])
        self.assertTrue(all(vue["rights"].values()))

    def test_un_autre_agent_ne_devient_pas_systeme_par_le_fichier(self):
        vue, _ = self.store.create("Robot")
        brut = json.loads((self.tmp / "agents.json").read_text("utf-8"))
        for a in brut["agents"]:
            if a["id"] == vue["id"]:
                a.update(system=True, rights={n: True for n in rights.RIGHTS})
        (self.tmp / "agents.json").write_text(json.dumps(brut), "utf-8")
        rouvert = keys.KeyStore(self.tmp).get(vue["id"])
        self.assertFalse(rouvert["system"])
        self.assertTrue(rouvert["rights"]["local_llm"])                                         # un droit ACCORDÉ dans le fichier reste accordé...
        self.assertFalse(keys.KeyStore(self.tmp).delete("aria"))                                # ...mais n'ouvre aucun verrou

    def test_aucun_agent_cree_ne_prend_l_identifiant_aria(self):
        autres = keys.KeyStore(Path(tempfile.mkdtemp(prefix="rights_b_")))
        autres.items = [k for k in autres.items if k["id"] != "aria"]                           # comme si l'agent par défaut n'existait pas encore
        for _ in range(30):
            vue, _ = autres.create("Robot")
            self.assertNotEqual(vue["id"], "aria")
            self.assertFalse(vue["system"])
        shutil.rmtree(autres.dir, ignore_errors=True)

    def test_les_champs_historiques_lus_bruts_sont_vrais_pour_l_agent_systeme(self):
        """api.py et sessions.py lisent `rec.get("publish_speaker")` sans passer par rights.has : l'enregistrement fourni par `authenticate` doit donc être vrai."""
        brut = json.loads((self.tmp / "agents.json").read_text("utf-8"))
        for a in brut["agents"]:
            if a["id"] == "aria":
                a.update(publish_links=False, publish_speaker=False)
        (self.tmp / "agents.json").write_text(json.dumps(brut), "utf-8")
        rouvert = keys.KeyStore(self.tmp)
        statut, rec = rouvert.authenticate(self.store.system_key)
        self.assertEqual(statut, "ok")
        self.assertTrue(rec["publish_links"] and rec["publish_speaker"])
        self.assertTrue(rouvert.record("aria")["publish_speaker"])

    def test_aucun_identifiant_genere_n_est_aria_meme_par_malchance(self):
        """Le tirage `k_xxxxxx` ne peut pas donner `aria`, mais la garde doit exister : on force le tirage."""
        import unittest.mock as mock
        with mock.patch.object(keys.secrets, "token_hex", side_effect=["aria"[:6]] + ["abcdef"] * 10):
            self.store.items = [k for k in self.store.items if k["id"] != "aria"]                 # l'identifiant `aria` est libre ; le préfixe `k_` l'en protège
            vue, _ = self.store.create("Robot")
        self.assertTrue(vue["id"].startswith("k_"))
        self.assertNotEqual(vue["id"], rights.SYSTEM_ID)

    def test_la_cle_de_l_agent_systeme_survit_a_un_redemarrage_sans_etre_recreee(self):
        """`ensure_system_agent` ne doit pas régénérer la clé à chaque démarrage : un client externe la garde."""
        cle = self.store.system_key
        for _ in range(3):
            rouvert = keys.KeyStore(self.tmp)
            self.assertEqual(rouvert.system_key, "")
            self.assertEqual(rouvert.authenticate(cle)[0], "ok")
        self.assertEqual(len([a for a in self.fichier() if a["id"] == "aria"]), 1)

    def test_ensure_system_agent_ne_recree_pas_et_retourne_faux(self):
        self.assertFalse(self.store.ensure_system_agent())
        avant = self.store.system_key
        self.store.ensure_system_agent()
        self.assertEqual(self.store.system_key, avant)

    def test_creer_un_agent_system_par_l_api_est_impossible(self):
        vue, _ = self.store.create("Faux système", rights_={n: True for n in rights.RIGHTS})
        self.assertFalse(vue["system"])
        self.assertFalse(self.store.delete("aria") and False)
        self.assertTrue(self.store.delete(vue["id"]))                                           # un agent ordinaire, même plein de droits, se supprime

    def test_create_ne_peut_pas_ecraser_un_agent_existant(self):
        vue, cle = self.store.create("Doublon", _kid="aria")
        self.assertIsNone(vue)
        self.assertEqual(self.store.authenticate(self.store.system_key)[0], "ok")


class DroitsDesAutresAgentsTests(Base):
    def test_defaut_aucun_droit(self):
        vue, _ = self.store.create("Robot")
        self.assertEqual(vue["rights"]["local_llm"], False)
        self.assertFalse(any(vue["rights"].values()))

    def test_accordes_a_la_creation_et_modifiables(self):
        vue, _ = self.store.create("Robot", rights_={"local_llm": True, "no_limits": True, "inconnu": True})
        self.assertEqual([n for n, v in vue["rights"].items() if v], ["local_llm", "no_limits"])
        vue = self.store.update(vue["id"], rights={"all_mcp": True})
        self.assertEqual([n for n, v in vue["rights"].items() if v], ["all_mcp"])
        vue = self.store.update(vue["id"], rights={})
        self.assertFalse(any(vue["rights"].values()))

    def test_les_deux_droits_historiques_sont_dans_la_vue_des_droits(self):
        vue, _ = self.store.create("Robot", publish_links=True, publish_speaker=True)
        self.assertTrue(vue["rights"]["publish_links"] and vue["rights"]["publish_speaker"])
        self.assertTrue(vue["publish_links"] and vue["publish_speaker"])                         # champs de premier niveau inchangés

    def test_les_droits_persistent_et_une_ancienne_cle_n_en_a_aucun(self):
        vue, _ = self.store.create("Robot", rights_={"all_mcp": True})
        self.assertTrue(keys.KeyStore(self.tmp).get(vue["id"])["rights"]["all_mcp"])
        brut = json.loads((self.tmp / "agents.json").read_text("utf-8"))
        for a in brut["agents"]:
            a.pop("rights", None)
            a.pop("system", None)
        (self.tmp / "agents.json").write_text(json.dumps(brut), "utf-8")
        rouvert = keys.KeyStore(self.tmp)
        self.assertFalse(any(rouvert.get(vue["id"])["rights"].values()))                         # aucune clé existante ne gagne de privilège
        self.assertTrue(rouvert.get("aria")["system"])

    def test_valeurs_non_booleennes_refusees_a_la_modification(self):
        vue, _ = self.store.create("Robot")
        for absurde in ({"local_llm": "true"}, {"local_llm": 1}, {"local_llm": [True]}, "local_llm", ["local_llm"], 5, None):
            apres = self.store.update(vue["id"], rights=absurde)
            self.assertFalse(any(apres["rights"].values()), repr(absurde))

    def test_modifier_un_autre_agent_ne_touche_pas_a_aria(self):
        vue, _ = self.store.create("Robot")
        self.store.update(vue["id"], rights={"local_llm": True}, label="Autre")
        self.assertTrue(all(self.store.get("aria")["rights"].values()))

    def test_record_expose_les_droits_effectifs_sans_le_hash(self):
        rec = self.store.record("aria")
        self.assertNotIn("hash", rec)
        self.assertTrue(all(rec["rights_effective"].values()))
        self.assertIsNone(self.store.record("inconnu"))
        vue, _ = self.store.create("Robot", rights_={"no_limits": True})
        self.assertEqual([n for n, v in self.store.record(vue["id"])["rights_effective"].items() if v], ["no_limits"])


if __name__ == "__main__":
    unittest.main()
