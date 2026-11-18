"""Migration « accès tiers » -> « agents » (components/agents/compat.py) : la configuration et les données de l'ancien format se lisent, ne se perdent
jamais, et s'écrivent au nouveau format. Bibliothèque standard seulement."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.agents import compat, keys, profile  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_lecture_de_l_ancienne_section(self):
        self.assertEqual(compat.section({"access": {"enabled": True}}), {"enabled": True})
        self.assertEqual(compat.section({"agents": {"enabled": False}}), {"enabled": False})
        self.assertEqual(compat.section({}), {})
        self.assertEqual(compat.section(None), {})

    def test_la_nouvelle_section_l_emporte_et_l_ancienne_n_est_pas_fusionnee(self):
        c = {"access": {"x": 1}, "agents": {"y": 2}}
        self.assertEqual(compat.section(c), {"y": 2})
        self.assertFalse(compat.migrate_config(c))
        self.assertEqual(c["access"], {"x": 1})                                  # rien n'est perdu

    def test_valeurs_invalides_ignorees(self):
        for v in (None, [], "x", 3, True):
            self.assertEqual(compat.section({"agents": v}), {})
            self.assertEqual(compat.section({"access": v}), {})
        self.assertEqual(compat.section({"agents": "x", "access": {"ok": 1}}), {"ok": 1})       # une valeur absurde ne masque pas l'ancienne

    def test_migration_en_memoire(self):
        c = {"access": {"enabled": True, "profiles": {"p": {}}}, "autre": 1}
        self.assertTrue(compat.migrate_config(c))
        self.assertEqual(c, {"agents": {"enabled": True, "profiles": {"p": {}}}, "autre": 1})
        self.assertFalse(compat.migrate_config(c))                               # idempotent

    def test_ecriture_sous_le_nouveau_nom_sans_garder_l_ancien(self):
        c = {"access": {"a": 1}}
        compat.write_section(c, {"a": 2})
        self.assertEqual(c, {"agents": {"a": 2}})

    def test_les_profils_se_lisent_et_s_ecrivent_dans_les_deux_formats(self):
        for nom in ("access", "agents"):
            c = {nom: {"profiles": {"demo": {"description": "D"}}}, "mcp": {"params": {"servers": {}}}}
            self.assertEqual(profile.ConfigProfiles(lambda c=c: c).names(), ["demo"], nom)
        c = {"access": {"profiles": {}, "private_servers": ["x"]}, "mcp": {"params": {"servers": {}}}}
        ok, err, changed, _ = profile.apply_profiles(c, json.dumps({"neuf": {"description": "N"}}))
        self.assertTrue(ok, err)
        self.assertNotIn("access", c)
        self.assertEqual(c["agents"]["private_servers"], ["x"])                  # les autres clés de la section sont conservées
        self.assertEqual(list(c["agents"]["profiles"]), ["neuf"])

    def test_serveurs_prives_et_hebergement_lus_dans_les_deux_formats(self):
        for nom in ("access", "agents"):
            c = {nom: {"private_servers": ["secret"], "hosted_only": ["m2"]}}
            self.assertEqual(profile.private_servers(c), {"secret"}, nom)
            self.assertEqual(profile.hosted_only(c), ["m2"], nom)


class DossierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="compat_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ancien(self, contenu="a"):
        d = self.tmp / "models" / "api_access"
        d.mkdir(parents=True)
        (d / "keys.json").write_text(contenu, "utf-8")
        return d

    def test_dossier_ancien_renomme_une_fois_avec_son_contenu(self):
        self.ancien("donnees")
        d = compat.data_dir(self.tmp)
        self.assertEqual(d, self.tmp / "models" / "agents")
        self.assertEqual((d / "keys.json").read_text("utf-8"), "donnees")
        self.assertFalse((self.tmp / "models" / "api_access").exists())
        self.assertEqual(compat.data_dir(self.tmp), d)                           # idempotent

    def test_les_deux_dossiers_existent_le_nouveau_gagne_et_l_ancien_reste(self):
        self.ancien("ancien")
        nouveau = self.tmp / "models" / "agents"
        nouveau.mkdir(parents=True)
        (nouveau / "agents.json").write_text("nouveau", "utf-8")
        self.assertEqual(compat.data_dir(self.tmp), nouveau)
        self.assertEqual((self.tmp / "models" / "api_access" / "keys.json").read_text("utf-8"), "ancien")
        self.assertEqual((nouveau / "agents.json").read_text("utf-8"), "nouveau")

    def test_ancienne_valeur_par_defaut_ecrite_dans_la_config_est_migree_aussi(self):
        self.ancien()
        self.assertEqual(compat.data_dir(self.tmp, "models/api_access"), self.tmp / "models" / "agents")

    def test_chemin_choisi_explicitement_est_respecte(self):
        self.ancien()
        self.assertEqual(compat.data_dir(self.tmp, "donnees/mes_agents"), self.tmp / "donnees" / "mes_agents")
        self.assertTrue((self.tmp / "models" / "api_access").is_dir())            # rien n'est déplacé

    def test_sans_aucun_dossier(self):
        self.assertEqual(compat.data_dir(self.tmp), self.tmp / "models" / "agents")


class FichierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="compat_f_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_un_ancien_keys_json_est_lu_puis_converti_a_la_premiere_ecriture(self):
        store = keys.KeyStore(self.tmp)
        vue, cle = store.create("Robot", ["https://a.example.com"], profile="p", publish_speaker=True)
        agents = json.loads((self.tmp / "agents.json").read_text("utf-8"))
        # on reconstitue l'ancien format : keys.json avec la clé « keys », et plus d'agents.json
        (self.tmp / "keys.json").write_text(json.dumps({"keys": agents["agents"]}), "utf-8")
        (self.tmp / "agents.json").unlink()
        rouvert = keys.KeyStore(self.tmp)
        self.assertEqual(rouvert.get(vue["id"])["label"], "Robot")
        self.assertTrue(rouvert.get(vue["id"])["publish_speaker"])
        self.assertEqual(rouvert.authenticate(cle, "https://a.example.com")[0], "ok")      # la clé API existante fonctionne toujours
        rouvert.update(vue["id"], label="Robot 2")                                          # première écriture : nouveau format
        self.assertTrue((self.tmp / "agents.json").exists())
        self.assertIn("agents", json.loads((self.tmp / "agents.json").read_text("utf-8")))
        self.assertFalse((self.tmp / "keys.json").exists())
        self.assertTrue((self.tmp / "keys.json.migre").exists())                           # jamais supprimé

    def test_le_nouveau_format_prime_et_l_ancien_est_conserve(self):
        (self.tmp / "keys.json").write_text(json.dumps({"keys": [{"id": "k_old", "hash": "x", "label": "Ancien"}]}), "utf-8")
        store = keys.KeyStore(self.tmp)
        store.create("Neuf")
        self.assertEqual(sorted(a["label"] for a in store.list()), ["Ancien", "Aria", "Neuf"])
        rouvert = keys.KeyStore(self.tmp)
        self.assertEqual(sorted(a["label"] for a in rouvert.list()), ["Ancien", "Aria", "Neuf"])

    def test_une_sauvegarde_migre_existante_n_est_jamais_ecrasee(self):
        """Si keys.json réapparaît (restauration d'une vieille sauvegarde) alors que keys.json.migre existe déjà, aucun des deux n'est perdu."""
        (self.tmp / "keys.json.migre").write_text("PREMIERE SAUVEGARDE", "utf-8")
        (self.tmp / "keys.json").write_text(json.dumps({"keys": [{"id": "k_r", "hash": "x", "label": "Restauré"}]}), "utf-8")
        store = keys.KeyStore(self.tmp)
        store.create("Neuf")                                                               # première écriture : tente de mettre keys.json de côté
        self.assertEqual((self.tmp / "keys.json.migre").read_text("utf-8"), "PREMIERE SAUVEGARDE")
        self.assertTrue((self.tmp / "keys.json").exists())                                 # resté en place, non supprimé

    def test_fichier_corrompu_ou_vide_ne_plante_pas(self):
        for contenu in ("", "{", "[]", '{"agents": "x"}', '{"keys": 3}'):
            (self.tmp / "agents.json").write_text(contenu, "utf-8")
            self.assertEqual([k["id"] for k in keys.KeyStore(self.tmp).list()], ["aria"], contenu)

    def test_le_secret_et_les_cles_survivent_au_changement_de_dossier(self):
        ancien = self.tmp / "models" / "api_access"
        store = keys.KeyStore(ancien)
        vue, cle = store.create("Robot", ["https://a.example.com"])
        nouveau = compat.data_dir(self.tmp)
        rouvert = keys.KeyStore(nouveau)
        self.assertEqual(rouvert.authenticate(cle, "https://a.example.com")[0], "ok")        # même secret.key, même clé API


if __name__ == "__main__":
    unittest.main()
