"""Liaison des accès tiers avec le serveur de mémoire unique (components/access/memory_link.py) : secret du canal, identité, serveurs résolus.
Bibliothèque standard seulement."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.access import memory_link, profile, sessions  # noqa: E402

MEM = {"run": "uv", "args": ["run", "/aria/tools/memory/server.py"], "env": {"MEMORY_DIR": "/aria/models/memory", "MEMORY_HTTP_PORT": "7872"}}
STDIO = {"run": "uv", "args": ["run", "/aria/tools/memory/server.py"], "env": {"MEMORY_DIR": "/aria/models/autre"}}
METEO = {"run": "uv", "args": ["run", "/aria/tools/weather_server.py"], "env": {"API_KEY": "<CleMeteo>"}}
SECRET = "un-secret-de-test-assez-long-0123456789"


class SecretTests(unittest.TestCase):
    def test_un_secret_neuf_a_chaque_appel_et_assez_long(self):
        a = memory_link.new_secret()
        b = memory_link.new_secret()
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(b), 16)
        self.assertEqual(memory_link.current_secret(), b)

    def test_le_secret_n_est_jamais_dans_une_configuration(self):
        memory_link.new_secret()
        c = memory_link.with_secret(MEM, memory_link.current_secret())
        self.assertNotIn("MEMORY_HTTP_SECRET", MEM["env"])                         # la configuration d'origine n'est pas modifiée
        self.assertEqual(c["env"]["MEMORY_HTTP_SECRET"], memory_link.current_secret())
        self.assertEqual(c["env"]["MEMORY_DIR"], MEM["env"]["MEMORY_DIR"])


class ConfigurationAriaTests(unittest.TestCase):
    def test_seul_un_serveur_avec_canal_http_recoit_le_secret(self):
        self.assertIn("MEMORY_HTTP_SECRET", memory_link.with_secret(MEM, SECRET)["env"])
        for autre in (STDIO, METEO):
            self.assertEqual(memory_link.with_secret(autre, SECRET), autre)

    def test_sans_secret_la_configuration_est_inchangee(self):
        memory_link.SECRET["value"] = ""
        self.assertEqual(memory_link.with_secret(MEM), MEM)

    def test_port(self):
        self.assertEqual(memory_link.http_port(MEM), 7872)
        for faux in (STDIO, METEO, {}, None, {"env": {"MEMORY_HTTP_PORT": "abc"}}, {"env": {"MEMORY_HTTP_PORT": "0"}}, {"env": {"MEMORY_HTTP_PORT": "70000"}}):
            self.assertIsNone(memory_link.http_port(faux), faux)


class SessionTiersTests(unittest.TestCase):
    def test_connexion_http_locale_avec_identite_et_secret(self):
        c = memory_link.third_party_conf(MEM, "k_robot1", "Bamboo 4WD", SECRET)
        self.assertEqual(c["url"], "http://127.0.0.1:7872/mcp")
        self.assertEqual(c["headers"]["Authorization"], f"Bearer {SECRET}")
        self.assertEqual(c["headers"]["X-Memory-Principal"], "k_robot1")
        self.assertEqual(c["headers"]["X-Memory-Name"], "Bamboo%204WD")
        self.assertNotIn("run", c)                                                 # aucun processus propre au tiers
        self.assertEqual(c["hide_tools"], ["identifier_interlocuteur"])

    def test_les_en_tetes_sont_de_l_ascii(self):
        c = memory_link.third_party_conf(MEM, "k_cam", "Caméra du garage « test »", SECRET)
        for valeur in c["headers"].values():
            valeur.encode("ascii")

    def test_pas_de_connexion_sans_secret_ni_identite(self):
        memory_link.SECRET["value"] = ""
        self.assertIsNone(memory_link.third_party_conf(MEM, "k_robot1", "Robot"))
        self.assertIsNone(memory_link.third_party_conf(MEM, "", "Robot", SECRET))
        self.assertIsNone(memory_link.third_party_conf(MEM, "k_robot1", "  ", SECRET))
        self.assertIsNone(memory_link.third_party_conf(STDIO, "k_robot1", "Robot", SECRET))        # pas de canal HTTP

    def test_resolution_des_serveurs_d_un_profil(self):
        out, manquants = memory_link.resolve_servers({"memoire": MEM, "meteo": METEO}, "k_r", "Robot", SECRET)
        self.assertEqual(manquants, [])
        self.assertIn("url", out["memoire"])
        self.assertEqual(out["meteo"], METEO)                                      # les autres serveurs sont inchangés
        self.assertNotIn("url", out["meteo"])

    def test_memoire_indisponible_signalee_jamais_ouverte_sans_authentification(self):
        out, manquants = memory_link.resolve_servers({"memoire": MEM}, "k_r", "Robot", "")
        memory_link.SECRET["value"] = ""
        out, manquants = memory_link.resolve_servers({"memoire": MEM}, "k_r", "Robot")
        self.assertEqual((out, manquants), ({}, ["memoire"]))

    def test_deux_acces_deux_identites_meme_serveur(self):
        a = memory_link.third_party_conf(MEM, "k_a", "A", SECRET)
        b = memory_link.third_party_conf(MEM, "k_b", "B", SECRET)
        self.assertEqual(a["url"], b["url"])                                       # un seul processus
        self.assertNotEqual(a["headers"]["X-Memory-Principal"], b["headers"]["X-Memory-Principal"])


class MoteurTests(unittest.TestCase):
    CONFIG = {"active_modules": {"llm": "A"}, "memory": {},
              "llm": {"A": {"params": {"backend_type": "anthropic", "model_name": "m", "apikey": "sk-test"}}},
              "mcp": {"params": {"servers": {"memoire": MEM}}}}

    def build(self, identity=("k_robot1", "Robot")):
        profil = {"name": "Robot", "description": "", "system_message": "R.", "llm": "", "voice": True, "mcp": {"servers": {"memoire": MEM}}}
        try:
            return sessions.build_engine(profil, self.CONFIG, None, identity=identity)
        except ImportError:                                                        # anthropic absent de l'hôte : testé dans le conteneur
            self.skipTest("anthropic absent")

    def test_le_moteur_d_une_session_se_connecte_au_canal_http(self):
        memory_link.new_secret()
        llm, mcp = self.build()
        conf = mcp.servers_conf["memoire"]
        self.assertEqual(conf["url"], "http://127.0.0.1:7872/mcp")
        self.assertEqual(conf["headers"]["X-Memory-Principal"], "k_robot1")

    def test_sans_secret_la_session_est_refusee(self):
        memory_link.SECRET["value"] = ""
        with self.assertRaises(profile.ProfileError) as cm:
            self.build()
        self.assertIn("mémoire", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
