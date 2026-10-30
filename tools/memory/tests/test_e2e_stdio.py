"""Test de bout en bout par le protocole MCP réel (stdio) : aucun code d'Aria n'est nécessaire.

Lancer : python -m unittest discover tools/memory/tests   (nécessite le paquet `mcp`, sinon le test est ignoré)
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

SERVER = Path(__file__).resolve().parent.parent / "server.py"


async def session_run(data_dir, calls):
    env = {**os.environ, "MEMORY_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            results = []
            for name, args in calls:
                res = await session.call_tool(name, args)
                results.append(res.content[0].text if res.content else "")
            return tools, results


@unittest.skipUnless(HAVE_MCP, "paquet mcp absent")
class E2ETests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="meme2e_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_outils_exposes_courts_et_plats(self):
        tools, _ = asyncio.run(session_run(self.tmp, []))
        self.assertEqual(sorted(t.name for t in tools), ["memoriser", "oublier", "rechercher_souvenir"])
        for tool in tools:
            self.assertLess(len(tool.description or ""), 200, tool.name)
            for prop in tool.inputSchema.get("properties", {}).values():
                self.assertIn(prop.get("type"), ("string", "integer"))        # schémas plats : pas de tableaux ni d'objets

    def test_aller_retour_et_persistance(self):
        _, first = asyncio.run(session_run(self.tmp, [
            ("memoriser", {"fait": "Marion préfère 19 degrés dans sa chambre", "type": "preference"}),
            ("memoriser", {"fait": "Marion préfère 19 degrés dans sa chambre", "type": "preference"}),
            ("rechercher_souvenir", {"question": "quelle température préfère Marion ?"}),
        ]))
        self.assertTrue(first[0].startswith("OK m_"))
        self.assertIn("Déjà connu", first[1])
        self.assertIn("Marion préfère 19 degrés", first[2])
        rid = first[0].split()[1]

        # Nouveau processus : les souvenirs doivent être relus depuis le disque
        _, second = asyncio.run(session_run(self.tmp, [
            ("rechercher_souvenir", {"question": "Marion"}),
            ("oublier", {"id": rid}),
            ("rechercher_souvenir", {"question": "Marion"}),
            ("oublier", {"id": "m_inconnu"}),
        ]))
        self.assertIn(rid, second[0])
        self.assertEqual(second[1], "Oublié.")
        self.assertEqual(second[2], "Aucun souvenir.")
        self.assertEqual(second[3], "Souvenir introuvable.")

        activity = (self.tmp / "logs" / "activity.jsonl").read_text("utf-8").splitlines()
        events = [json.loads(l)["ev"] for l in activity]
        self.assertIn("add", events)
        self.assertIn("search", events)
        self.assertIn("delete", events)
        archive = list((self.tmp / "default" / "archive").glob("*.jsonl"))
        self.assertTrue(archive and "Marion" in archive[0].read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
