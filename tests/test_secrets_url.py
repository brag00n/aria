"""Résolution des balises <NOM> dans les URL des moteurs LLM (base_url). Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import components.utils as utils
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes")
class ExpandUrlTests(unittest.TestCase):
    def setUp(self):
        os.environ["T_URL_X"] = "http://10.0.0.5:1234/v1"

    def tearDown(self):
        os.environ.pop("T_URL_X", None)

    def test_balise_resolue(self):
        self.assertEqual(utils.expand_url("<T_URL_X>"), "http://10.0.0.5:1234/v1")
        self.assertEqual(utils.expand_url("<T_URL_X>/chat"), "http://10.0.0.5:1234/v1/chat")

    def test_balise_sans_variable_donne_vide(self):
        self.assertEqual(utils.expand_url("<T_URL_ABSENTE>"), "")
        self.assertEqual(utils.expand_url("http://h/<T_URL_ABSENTE>"), "")

    def test_url_en_clair_et_vide(self):
        self.assertEqual(utils.expand_url("http://h:1/v1"), "http://h:1/v1")
        self.assertEqual(utils.expand_url(None), "")
        self.assertEqual(utils.expand_url(""), "")

    def test_moteurs_utilisent_expand_url(self):
        for f in ("llm_connect_anthropic.py", "llm_connect_openai.py"):
            src = (Path(__file__).resolve().parent.parent / "components" / f).read_text("utf-8")
            self.assertIn('utils.expand_url(self.params.get("base_url")', src, f)


if __name__ == "__main__":
    unittest.main()
