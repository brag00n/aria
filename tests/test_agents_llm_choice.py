"""Chaque agent (via son profil) choisit son module LLM ; deux modules du même type avec clé et modèle différents donnent deux moteurs distincts.
Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from components.agents.sessions import build_engine
    from components.agents import profile as P
    HAVE = True
except ImportError:
    HAVE = False

CFG = {"llm": {
    "Llm_A": {"params": {"backend_type": "anthropic", "apikey": "<T_KEY_A>", "model_name": "modele-a", "base_url": "<T_URL_A>"}},
    "Llm_B": {"params": {"backend_type": "anthropic", "apikey": "<T_KEY_B>", "model_name": "modele-b"}},
    "Llm_O": {"params": {"backend_type": "openai", "apikey": "<T_KEY_O>", "model_name": "modele-o", "base_url": "http://o.test/v1"}}},
    "active_modules": {"llm": "Llm_A"}}


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes")
class ChoixLlmTests(unittest.TestCase):
    def setUp(self):
        os.environ.update(T_KEY_A="cle-a", T_KEY_B="cle-b", T_KEY_O="cle-o", T_URL_A="http://a.test")

    def tearDown(self):
        for k in ("T_KEY_A", "T_KEY_B", "T_KEY_O", "T_URL_A"):
            os.environ.pop(k, None)

    def moteur(self, nom):
        llm, _ = build_engine({"llm": nom, "mcp": {"servers": {}}, "system_message": "x"}, CFG)
        return llm.engine

    def test_deux_modules_anthropic_deux_cles_deux_modeles(self):
        a, b = self.moteur("Llm_A"), self.moteur("Llm_B")
        self.assertEqual((a.model_name, a.apikey, a.base_url), ("modele-a", "cle-a", "http://a.test"))
        self.assertEqual((b.model_name, b.apikey, b.base_url), ("modele-b", "cle-b", None))

    def test_la_cle_openai_est_resolue(self):
        o = self.moteur("Llm_O")
        self.assertEqual((o.model_name, o.apikey), ("modele-o", "cle-o"))

    def test_sans_llm_le_module_actif_d_aria(self):
        self.assertEqual(self.moteur("").model_name, "modele-a")

    def test_validation_d_un_module_inconnu(self):
        self.assertEqual(P.validate_profiles({"p": {"llm": "Llm_B"}}, CFG), [])
        self.assertTrue(P.validate_profiles({"p": {"llm": "Inconnu"}}, CFG))


if __name__ == "__main__":
    unittest.main()
