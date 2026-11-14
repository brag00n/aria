"""Sauvegarde de l'onglet Paramètres (components/settings.py) : validation de tout avant modification, profils d'accès, écriture atomique en UTF-8."""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import settings  # noqa: E402


def base_config():
    return {
        "global": {"language": "fr", "default_language": "en"},
        "conversation": {"enabled": True, "open_phrases": ["bonjour aria"], "close_phrases": ["merci aria"], "stop_phrases": ["stop"]},
        "memory": {"max_exchanges": 8, "max_chars": 12000, "tool_result_max_chars": 1500},
        "active_modules": {"vad": "Vad_Web", "stt": "Stt_A", "llm": "Llm_Anthropic", "tts": "Tts_A"},
        "vad": {"Vad_Web": {"params": {"positive_threshold": 0.8}}},
        "stt": {"Stt_A": {"params": {"model_name": "small"}}, "Stt_B": {"params": {}}},
        "llm": {"Llm_Anthropic": {"params": {"backend_type": "anthropic", "model_name": "claude-haiku-5-5"}},
                "Llm_Local": {"params": {"backend_type": "local"}}},
        "tts": {"Tts_A": {"params": {"voice": "x"}}},
        "mcp": {"params": {"servers": {
            "meteo": {"run": "uv", "args": ["run", "m.py"], "env": {"API_KEY": "<CleMeteo>"}},
            "web": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer <Jeton>"}},
            "memoire": {"run": "uv", "args": ["run", "mem.py"]}}}},
        "access": {"enabled": True, "port": 7861, "profiles": {"demo": {"description": "Démo", "servers": ["meteo"]}}},
    }


def fields_of(cfg, **override):
    f = {"global": json.dumps(cfg["global"]), "conversation": json.dumps(cfg["conversation"]), "memory": json.dumps(cfg["memory"]),
         "mcp": json.dumps(cfg["mcp"]["params"]), "profiles": json.dumps(cfg["access"].get("profiles", {})),
         "vad_choice": cfg["active_modules"]["vad"], "stt_choice": cfg["active_modules"]["stt"],
         "llm_choice": cfg["active_modules"]["llm"], "tts_choice": cfg["active_modules"]["tts"]}
    for cat in ("vad", "stt", "llm", "tts"):
        f[cat] = json.dumps(cfg[cat][cfg["active_modules"][cat]]["params"])
    f.update(override)
    return f


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def apply(self, **override):
        return settings.apply_settings(self.cfg, fields_of(self.cfg, **override))

    def test_sauvegarde_sans_changement(self):
        avant = copy.deepcopy(self.cfg)
        parsed, changed = self.apply()
        self.assertEqual(changed, [])
        self.assertEqual(self.cfg, avant)

    def test_aucune_modification_quand_un_champ_est_invalide(self):
        avant = copy.deepcopy(self.cfg)
        invalides = [
            {"global": "{pas du json"}, {"global": "[1]"}, {"global": json.dumps({"language": "zzzzz"})},
            {"conversation": json.dumps({"open_phrases": "bonjour"})}, {"memory": json.dumps({"max_exchanges": -1})},
            {"memory": json.dumps({"max_chars": True})},
            {"vad": "{x"}, {"stt": "[]"}, {"llm": "pas du json"}, {"tts": "3"}, {"mcp": "{x"}, {"mcp": "[]"},
            {"profiles": "{x"}, {"profiles": "[1]"}, {"profiles": json.dumps({"a": {"servers": ["memoire"]}})},
            {"profiles": json.dumps({"a": {"servers": ["inconnu"]}})}, {"stt_choice": "Stt_Inconnu"},
        ]
        for champ in invalides:
            with self.assertRaises((ValueError, json.JSONDecodeError), msg=str(champ)):
                self.apply(**champ)
            self.assertEqual(self.cfg, avant, f"config modifiée malgré l'erreur : {champ}")

    def test_erreur_dans_un_champ_tardif_ne_laisse_rien_de_modifie(self):
        # Avant : global/conversation/memory étaient écrits puis une erreur sur mcp laissait la config à moitié modifiée
        avant = copy.deepcopy(self.cfg)
        with self.assertRaises(json.JSONDecodeError):
            self.apply(**{"global": json.dumps({"language": "en"}), "mcp": "{pas du json"})
        self.assertEqual(self.cfg, avant)
        self.assertEqual(self.cfg["global"]["language"], "fr")

    def test_modification_valide_appliquee_partout(self):
        self.apply(**{"global": json.dumps({"language": "EN", "default_language": "fr"}), "stt_choice": "Stt_B", "stt": json.dumps({"model_name": "large"})})
        self.assertEqual((self.cfg["global"]["language"], self.cfg["global"]["default_language"]), ("en", "fr"))
        self.assertEqual(self.cfg["active_modules"]["stt"], "Stt_B")
        self.assertEqual(self.cfg["stt"]["Stt_B"]["params"], {"model_name": "large"})

    def test_les_autres_cles_de_access_sont_conservees(self):
        self.apply(profiles=json.dumps({"neuf": {"servers": ["web"]}}))
        self.assertEqual((self.cfg["access"]["enabled"], self.cfg["access"]["port"]), (True, 7861))
        self.assertEqual(list(self.cfg["access"]["profiles"]), ["neuf"])


class ProfilsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def apply(self, profiles):
        return settings.apply_settings(self.cfg, fields_of(self.cfg, profiles=json.dumps(profiles)))

    def test_noms_des_profils_modifies(self):
        _, changed = self.apply({"demo": {"description": "Démo", "servers": ["meteo"]}, "autre": {"servers": ["web"]}})
        self.assertEqual(changed, ["autre"])
        _, changed = self.apply({"autre": {"servers": ["web", "meteo"]}})
        self.assertEqual(changed, ["autre", "demo"])                                       # modifié et supprimé

    def test_un_profil_peut_utiliser_un_serveur_ajoute_dans_la_meme_sauvegarde(self):
        mcp = copy.deepcopy(self.cfg["mcp"]["params"])
        mcp["servers"]["nouveau"] = {"url": "http://n/mcp"}
        parsed, changed = settings.apply_settings(self.cfg, fields_of(self.cfg, mcp=json.dumps(mcp), profiles=json.dumps({"p": {"servers": ["nouveau"]}})))
        self.assertEqual(list(self.cfg["access"]["profiles"]["p"]["servers"]), ["nouveau"])

    def test_un_serveur_retire_dans_la_meme_sauvegarde_invalide_le_profil(self):
        mcp = copy.deepcopy(self.cfg["mcp"]["params"])
        del mcp["servers"]["meteo"]                                                         # « demo » le référence encore
        avant = copy.deepcopy(self.cfg)
        with self.assertRaises(ValueError) as cm:
            settings.apply_settings(self.cfg, fields_of(self.cfg, mcp=json.dumps(mcp)))
        self.assertIn("meteo", str(cm.exception))
        self.assertEqual(self.cfg, avant)

    def test_llm_local_refuse_dans_un_profil(self):
        with self.assertRaises(ValueError) as cm:
            self.apply({"a": {"llm": "Llm_Local"}})
        self.assertIn("local", str(cm.exception))

    def test_message_d_erreur_lisible_et_complet(self):
        with self.assertRaises(ValueError) as cm:
            self.apply({"a": {"servers": ["memoire", "fantome"], "voix": True}})
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("❌"))
        for mot in ("privé", "fantome", "voix"):
            self.assertIn(mot, msg)

    def test_texte_vide_supprime_les_profils(self):
        parsed, changed = settings.apply_settings(self.cfg, fields_of(self.cfg, profiles=""))
        self.assertEqual((self.cfg["access"]["profiles"], changed), ({}, ["demo"]))

    def test_les_secrets_des_serveurs_ne_sont_pas_touches(self):
        self.apply({"p": {"servers": ["meteo", "web"]}})
        servers = self.cfg["mcp"]["params"]["servers"]
        self.assertEqual(servers["meteo"]["env"], {"API_KEY": "<CleMeteo>"})                # tags conservés, jamais remplacés par des valeurs
        self.assertEqual(servers["web"]["headers"], {"Authorization": "Bearer <Jeton>"})
        self.assertNotIn("CleMeteo", json.dumps(self.cfg["access"]))                       # et jamais copiés dans les profils


class EcritureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="settings_"))
        self.path = self.tmp / "default.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ecriture_utf8_lisible(self):
        settings.write_config({"a": "Démonstration : météo, Aria sarcastique"}, str(self.path))
        brut = self.path.read_text("utf-8")
        self.assertIn("Démonstration : météo", brut)
        self.assertNotIn("\\u00e9", brut)                                                  # plus d'accents échappés
        self.assertEqual(json.loads(brut), {"a": "Démonstration : météo, Aria sarcastique"})

    def test_remplacement_atomique_aucun_fichier_temporaire_ne_reste(self):
        self.path.write_text('{"ancien": 1}', "utf-8")
        settings.write_config({"nouveau": 2}, str(self.path))
        self.assertEqual(json.loads(self.path.read_text("utf-8")), {"nouveau": 2})
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["default.json"])

    def test_un_echec_d_ecriture_laisse_l_ancien_fichier_intact(self):
        self.path.write_text('{"ancien": 1}', "utf-8")
        with self.assertRaises(TypeError):
            settings.write_config({"x": object()}, str(self.path))                           # non sérialisable : l'écriture échoue en cours de route
        self.assertEqual(json.loads(self.path.read_text("utf-8")), {"ancien": 1})
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["default.json"])             # le fichier temporaire est supprimé


class AideEtMessagesTests(unittest.TestCase):
    def test_aide_liste_serveurs_utilisables_et_prives(self):
        aide = settings.profiles_help(base_config())
        self.assertIn("meteo, web", aide)
        self.assertIn("jamais exposés : memoire", aide)

    def test_aide_sans_serveur(self):
        self.assertIn("aucun", settings.profiles_help({}))

    def test_message_de_sauvegarde(self):
        simple = settings.save_message([], 0, [])
        self.assertIn("sauvegardée", simple)
        self.assertNotIn("Profils d'accès", simple)
        complet = settings.save_message(["a", "b"], 2, ["clé X utilise le profil a"])
        for mot in ("a, b", "2 conversation(s)", "⚠️ clé X"):
            self.assertIn(mot, complet)


if __name__ == "__main__":
    unittest.main()
