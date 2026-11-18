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
            "memoire": {"run": "uv", "args": ["run", "tools/memory/server.py"], "env": {"MEMORY_DIR": "/m", "MEMORY_HTTP_PORT": "7872"}},
            "memoire_seule": {"run": "uv", "args": ["run", "tools/memory/server.py"], "env": {"MEMORY_DIR": "/m2"}}}}},
        "agents": {"enabled": True, "port": 7861, "profiles": {"demo": {"description": "Démo", "servers": ["meteo"]}}},
    }


def fields_of(cfg, **override):
    f = {"global": json.dumps(cfg["global"]), "conversation": json.dumps(cfg["conversation"]), "memory": json.dumps(cfg["memory"]),
         "mcp": json.dumps(cfg["mcp"]["params"]),
         "vad_choice": cfg["active_modules"]["vad"], "stt_choice": cfg["active_modules"]["stt"],
         "llm_choice": cfg["active_modules"]["llm"], "tts_choice": cfg["active_modules"]["tts"]}
    for cat in ("vad", "stt", "llm", "tts"):
        f[cat] = json.dumps(cfg[cat][cfg["active_modules"][cat]]["params"])
    f.update(override)
    return f


class AncienneSectionTests(unittest.TestCase):
    """Une config qui a encore la section « access » : lue telle quelle, puis réécrite sous « agents » par la sauvegarde des Paramètres."""

    def test_sauvegarde_d_une_ancienne_config(self):
        cfg = base_config()
        cfg["access"] = cfg.pop("agents")
        cfg["access"]["private_servers"] = ["secret"]
        settings.apply_settings(cfg, fields_of({**cfg, "agents": cfg["access"]}))
        self.assertEqual(list(cfg["access"]["profiles"]), ["demo"])                       # les Paramètres ne touchent plus aux profils ni à leur section
        self.assertEqual(cfg["access"]["private_servers"], ["secret"])


class AuditTests(unittest.TestCase):
    """Section « audit » : validée avant toute modification ; vide = inchangée."""

    def setUp(self):
        self.cfg = base_config()

    def apply(self, audit):
        return settings.apply_settings(self.cfg, fields_of(self.cfg, audit=audit))

    def test_section_vide_ou_absente_ne_change_rien(self):
        self.apply("")
        self.assertNotIn("audit", self.cfg)
        settings.apply_settings(self.cfg, fields_of(self.cfg))
        self.assertNotIn("audit", self.cfg)

    def test_section_valide_appliquee(self):
        self.apply(json.dumps({"enabled": True, "retention_days": 30, "log_text": False, "dir": "models/audit"}))
        self.assertEqual(self.cfg["audit"], {"enabled": True, "retention_days": 30, "log_text": False, "dir": "models/audit"})

    def test_valeurs_refusees_sans_rien_modifier(self):
        avant = copy.deepcopy(self.cfg)
        for faux in ({"retention_days": 0}, {"retention_days": -1}, {"retention_days": "90"}, {"retention_days": True}, {"retention_days": 99999},
                     {"retention_days": 1.5}, {"enabled": "oui"}, {"log_text": 1}, {"dir": "/etc"}, {"dir": "../dehors"}, {"dir": "a/../../b"},
                     {"dir": ""}, {"dir": 3}, {"inconnue": 1}):
            with self.assertRaises(ValueError, msg=str(faux)):
                self.apply(json.dumps(faux))
            self.assertEqual(self.cfg, avant, str(faux))
        for pas_objet in ("[1]", "3", "\"x\""):
            with self.assertRaises(ValueError):
                self.apply(pas_objet)
        with self.assertRaises(json.JSONDecodeError):
            self.apply("{x")

    def test_une_erreur_d_audit_n_applique_aucune_autre_section(self):
        avant = copy.deepcopy(self.cfg)
        with self.assertRaises(ValueError):
            settings.apply_settings(self.cfg, fields_of(self.cfg, memory=json.dumps({"max_exchanges": 3}), audit=json.dumps({"retention_days": 0})))
        self.assertEqual(self.cfg, avant)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def apply(self, **override):
        return settings.apply_settings(self.cfg, fields_of(self.cfg, **override))

    def test_sauvegarde_sans_changement(self):
        avant = copy.deepcopy(self.cfg)
        self.apply()
        self.assertEqual(self.cfg, avant)

    def test_aucune_modification_quand_un_champ_est_invalide(self):
        avant = copy.deepcopy(self.cfg)
        invalides = [
            {"global": "{pas du json"}, {"global": "[1]"}, {"global": json.dumps({"language": "zzzzz"})},
            {"conversation": json.dumps({"open_phrases": "bonjour"})}, {"memory": json.dumps({"max_exchanges": -1})},
            {"memory": json.dumps({"max_chars": True})},
            {"vad": "{x"}, {"stt": "[]"}, {"llm": "pas du json"}, {"tts": "3"}, {"mcp": "{x"}, {"mcp": "[]"},
            {"stt_choice": "Stt_Inconnu"},
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
        self.apply()
        self.assertEqual((self.cfg["agents"]["enabled"], self.cfg["agents"]["port"]), (True, 7861))
        self.assertEqual(list(self.cfg["agents"]["profiles"]), ["demo"])


class ProfilsTests(unittest.TestCase):
    """Les profils ne se modifient plus dans les Paramètres : la sauvegarde les laisse intacts et refuse une config qui en invaliderait un."""

    def setUp(self):
        self.cfg = base_config()

    def apply(self, **override):
        return settings.apply_settings(self.cfg, fields_of(self.cfg, **override))

    def test_les_profils_sont_conserves_tels_quels(self):
        avant = copy.deepcopy(self.cfg["agents"]["profiles"])
        self.apply()
        self.assertEqual(self.cfg["agents"]["profiles"], avant)

    def test_un_champ_profiles_est_ignore(self):
        self.apply(profiles=json.dumps({"intrus": {"servers": ["web"]}}))
        self.assertEqual(list(self.cfg["agents"]["profiles"]), ["demo"])

    def test_retirer_un_serveur_utilise_par_un_profil_est_refuse(self):
        mcp = copy.deepcopy(self.cfg["mcp"]["params"])
        del mcp["servers"]["meteo"]
        avant = copy.deepcopy(self.cfg)
        with self.assertRaises(ValueError) as cm:
            self.apply(mcp=json.dumps(mcp))
        self.assertIn("meteo", str(cm.exception))
        self.assertIn("profil", str(cm.exception))
        self.assertEqual(self.cfg, avant)

    def test_un_llm_local_reste_valide_dans_un_profil(self):
        self.cfg["agents"]["profiles"]["loc"] = {"llm": "Llm_Local"}
        self.apply()                                                       # le droit « LLM local » appartient à l'agent, pas à la sauvegarde
        from components.agents import profile as P
        pr = P.ConfigProfiles(lambda: self.cfg)
        with self.assertRaises(P.ProfileError):
            pr.load("loc")                                                 # un agent sans le droit est refusé au démarrage de sa session
        self.assertEqual(pr.load("loc", allow_local=True)["llm"], "Llm_Local")
        self.assertTrue(pr.describe("loc")["llm_local"])

    def test_les_secrets_des_serveurs_ne_sont_pas_touches(self):
        self.cfg["agents"]["profiles"]["p"] = {"servers": ["meteo", "web"]}
        self.apply()
        servers = self.cfg["mcp"]["params"]["servers"]
        self.assertEqual(servers["meteo"]["env"], {"API_KEY": "<CleMeteo>"})                # tags conservés, jamais remplacés par des valeurs
        self.assertEqual(servers["web"]["headers"], {"Authorization": "Bearer <Jeton>"})
        self.assertNotIn("CleMeteo", json.dumps(self.cfg["agents"]))                       # et jamais copiés dans les profils


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


class MessagesTests(unittest.TestCase):
    def test_message_de_sauvegarde(self):
        simple = settings.save_message([])
        self.assertIn("sauvegardée", simple)
        self.assertNotIn("Profils", simple)
        self.assertIn("⚠️ clé X", settings.save_message(["clé X invalide"]))


if __name__ == "__main__":
    unittest.main()
