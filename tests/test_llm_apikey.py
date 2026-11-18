"""La clé API d'un module LLM vient UNIQUEMENT de params.apikey : aucun repli sur une variable d'environnement (un agent mal configuré ne doit pas utiliser
la clé d'un autre module). Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import components.utils as utils
    from components.llm import Llm
    from components.agents.sessions import build_engine
    HAVE = True
except ImportError:
    HAVE = False

AUTRES = ("LLM_ANTHROPIC_DEFAULT_APIKEY", "ANTHROPIC_API_KEY", "LLM_OPENAI_DEFAULT_APIKEY", "OPENAI_API_KEY", "T_CLE_A")


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes")
class CleApiTests(unittest.TestCase):
    def setUp(self):
        self.sauve = {k: os.environ.get(k) for k in AUTRES}
        os.environ["LLM_ANTHROPIC_DEFAULT_APIKEY"] = "cle-d-un-autre"          # la variable qui servait de repli
        os.environ["ANTHROPIC_API_KEY"] = "cle-sdk"
        os.environ["OPENAI_API_KEY"] = "cle-sdk-openai"
        os.environ["T_CLE_A"] = "cle-a"

    def tearDown(self):
        for k, v in self.sauve.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def moteur(self, backend, **params):
        return Llm({"backend_type": backend, "model_name": "m", **params}).engine

    def test_la_cle_vient_du_module(self):
        self.assertEqual(self.moteur("anthropic", apikey="<T_CLE_A>").apikey, "cle-a")
        self.assertEqual(self.moteur("openai", apikey="<T_CLE_A>", base_url="http://x/v1").apikey, "cle-a")
        self.assertEqual(self.moteur("anthropic", apikey="en-clair").apikey, "en-clair")

    def test_anthropic_sans_cle_est_une_erreur_meme_avec_la_variable_de_repli(self):
        for p in ({}, {"apikey": ""}, {"apikey": None}, {"apikey": "   "}):
            with self.assertRaises(ValueError, msg=str(p)) as cm:
                self.moteur("anthropic", **p)
            self.assertIn("params.apikey", str(cm.exception))

    def test_balise_sans_variable_est_une_erreur_et_n_est_jamais_envoyee_comme_cle(self):
        with self.assertRaises(ValueError):
            self.moteur("anthropic", apikey="<T_CLE_ABSENTE>")
        with self.assertRaises(ValueError):
            self.moteur("openai", apikey="<T_CLE_ABSENTE>", base_url="http://x/v1")

    def test_openai_sans_cle_est_une_erreur(self):
        for p in ({}, {"apikey": ""}):
            with self.assertRaises(ValueError, msg=str(p)):
                self.moteur("openai", base_url="http://x/v1", **p)

    def test_serveurs_locaux_sans_authentification_gardent_leur_valeur_factice(self):
        self.assertEqual(self.moteur("llm_studio", base_url="http://x/v1").apikey, "llm_studio")
        self.assertEqual(self.moteur("ollama", base_url="http://x/v1").apikey, "ollama")
        self.assertEqual(self.moteur("ollama", base_url="http://x/v1", apikey="<T_CLE_A>").apikey, "cle-a")

    def test_la_cle_d_un_agent_n_est_jamais_celle_d_un_autre_module(self):
        cfg = {"llm": {"A": {"params": {"backend_type": "anthropic", "apikey": "<T_CLE_A>", "model_name": "m"}},
                       "B": {"params": {"backend_type": "anthropic", "model_name": "m"}}}, "active_modules": {"llm": "A"}}
        a = build_engine({"llm": "A", "mcp": {"servers": {}}, "system_message": "x"}, cfg)[0].engine
        self.assertEqual(a.apikey, "cle-a")
        with self.assertRaises(ValueError):                                      # B n'a pas de clé : erreur, pas la clé de A ni la variable de repli
            build_engine({"llm": "B", "mcp": {"servers": {}}, "system_message": "x"}, cfg)

    def test_expand_apikey(self):
        self.assertEqual(utils.expand_apikey("<T_CLE_A>"), "cle-a")
        self.assertEqual(utils.expand_apikey(None), "")
        self.assertEqual(utils.expand_apikey("<T_CLE_ABSENTE>"), "")
        self.assertEqual(utils.expand_apikey("pre-<T_CLE_ABSENTE>"), "")

    def test_plus_aucune_lecture_de_variable_de_repli_dans_le_code(self):
        racine = Path(__file__).resolve().parent.parent / "components"
        for f in ("llm_connect_anthropic.py", "llm_connect_openai.py", "llm.py"):
            src = (racine / f).read_text("utf-8")
            self.assertNotIn("os.environ.get(\"LLM_ANTHROPIC", src, f)
            self.assertNotIn("ANTHROPIC_API_KEY", src, f)


if __name__ == "__main__":
    unittest.main()
