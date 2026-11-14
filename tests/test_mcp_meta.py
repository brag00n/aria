"""`Mcp.call_tool` joint l'échange courant au `_meta` du message MCP (hors des arguments) et note l'outil utilisé. Session factice : aucun serveur."""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from components import echanges
    from components.mcp import Mcp
    HAVE = True
except ImportError:
    HAVE = False


class Contenu:
    def __init__(self, text):
        self.text = text


class Resultat:
    def __init__(self, text):
        self.content = [Contenu(text)]


class FausseSession:
    def __init__(self):
        self.appels = []

    async def call_tool(self, name, arguments=None, meta=None, **kw):
        self.appels.append({"name": name, "arguments": arguments, "meta": meta})
        await asyncio.sleep(0.01)
        return Resultat("OK")


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class MetaTests(unittest.TestCase):
    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.mcp = Mcp({"servers": {}})
        self.session = FausseSession()
        self.mcp.tool_registry["memoriser"] = self.session
        echanges.clear()

    def tearDown(self):
        echanges.clear()
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_l_echange_courant_part_dans_le_meta_et_pas_dans_les_arguments(self):
        async def tour():
            e = echanges.start("k_1:Robot", "Robot", "c", query="Retiens que j'aime le thé")
            await self.mcp.call_tool("memoriser", {"fait": "x"})
            return e
        e = asyncio.run(tour())
        (appel,) = self.session.appels
        self.assertEqual(appel["meta"]["echange"], e.id)
        self.assertEqual((appel["meta"]["src"], appel["meta"]["x"]), ("k_1:Robot", "Retiens que j'aime le thé"))
        self.assertEqual(appel["arguments"], {"fait": "x"})
        self.assertEqual(e.outils, ["memoriser"])

    def test_hors_echange_pas_de_meta(self):
        asyncio.run(self.mcp.call_tool("memoriser", {"fait": "x"}))
        self.assertIsNone(self.session.appels[0]["meta"])

    def test_le_llm_ne_peut_pas_glisser_un_meta_par_les_arguments(self):
        async def tour():
            echanges.start("aria", "Olivier", "c", query="q")
            await self.mcp.call_tool("memoriser", {"fait": "x", "_meta": {"echange": "e_falsifie123"}})
        asyncio.run(tour())
        self.assertNotEqual(self.session.appels[0]["meta"]["echange"], "e_falsifie123")

    def test_tours_concurrents_chacun_son_echange(self):
        async def tour(n):
            echanges.start(f"k_{n}:A{n}", f"A{n}", f"c{n}", query=f"q{n}")
            await asyncio.sleep(0.005 * (5 - n))
            await self.mcp.call_tool("memoriser", {"fait": str(n)})
            return echanges.current().id

        async def tous():
            return await asyncio.gather(*(tour(n) for n in range(1, 5)))
        ids = asyncio.run(tous())
        par_fait = {a["arguments"]["fait"]: a["meta"] for a in self.session.appels}
        for n, eid in zip(range(1, 5), ids):
            self.assertEqual((par_fait[str(n)]["echange"], par_fait[str(n)]["src"]), (eid, f"k_{n}:A{n}"))

    def test_outil_inconnu_ne_note_rien(self):
        async def tour():
            e = echanges.start("aria", "x", "c")
            await self.mcp.call_tool("inconnu", {})
            return e
        self.assertEqual(asyncio.run(tour()).outils, [])


if __name__ == "__main__":
    unittest.main()
