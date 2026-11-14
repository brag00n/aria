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
        # 5 outils pour le LLM + ui_etat et ui_effacer_traces, que mcp.py cache au LLM (préfixe ui_)
        self.assertEqual(sorted(t.name for t in tools), ["identifier_interlocuteur", "memoriser", "oublier", "rechercher_souvenir",
                                                         "relier", "ui_effacer_traces", "ui_etat"])
        for tool in tools:
            self.assertLess(len(tool.description or ""), 200, tool.name)
            for prop in tool.inputSchema.get("properties", {}).values():
                self.assertIn(prop.get("type"), ("string", "integer"))        # schémas plats : pas de tableaux ni d'objets

    def test_aller_retour_et_persistance(self):
        _, first = asyncio.run(session_run(self.tmp, [
            ("memoriser", {"fait": "Marion préfère 19 degrés", "type": "preference"}),         # interlocuteur inconnu : refusé
            ("identifier_interlocuteur", {"nom": "Olivier"}),
            ("memoriser", {"fait": "Marion préfère 19 degrés dans sa chambre", "type": "preference"}),
            ("memoriser", {"fait": "Marion préfère 19 degrés dans sa chambre", "type": "preference"}),
            ("rechercher_souvenir", {"question": "quelle température préfère Marion ?"}),
            ("relier", {"personne_a": "Tiphaine", "lien": "femme de", "personne_b": "moi"}),
            ("ui_etat", {"activer": "?"}),
        ]))
        self.assertIn("Interlocuteur inconnu", first[0])
        self.assertIn("créé et actif", first[1])
        self.assertTrue(first[2].startswith("OK m_"))
        self.assertIn("Déjà connu", first[3])
        self.assertIn("Marion préfère 19 degrés", first[4])
        self.assertIn("Tiphaine est la femme de Olivier", first[5])
        etat = json.loads(first[6])
        self.assertEqual((etat["actif"], etat["liste"]), ("p1", [{"id": "p1", "label": "Olivier"}]))
        rid = first[2].split()[1]

        # Nouveau processus : les souvenirs sont relus, mais l'interlocuteur actif retombe sur Inconnue (Aria le resynchronise)
        _, second = asyncio.run(session_run(self.tmp, [
            ("ui_etat", {"activer": "?"}),
            ("rechercher_souvenir", {"question": "Marion"}),            # Inconnue : le privé est invisible
            ("rechercher_souvenir", {"question": "Tiphaine"}),          # la relation est partagée
            ("ui_etat", {"activer": "p1"}),
            ("rechercher_souvenir", {"question": "Marion"}),
            ("oublier", {"id": rid}),
            ("rechercher_souvenir", {"question": "Marion"}),
            ("oublier", {"id": "m_inconnu"}),
        ]))
        self.assertEqual(json.loads(second[0])["actif"], "")
        self.assertNotIn("Marion préfère", second[1])                  # le privé d'Olivier reste invisible (le repli ne montre que le partagé)
        self.assertIn("femme de", second[2])
        self.assertEqual(json.loads(second[3])["actif"], "p1")
        self.assertIn(rid, second[4])
        self.assertEqual(second[5], "Oublié.")
        self.assertNotIn("Marion préfère", second[6])
        self.assertEqual(second[7], "Souvenir introuvable.")

        activity = (self.tmp / "logs" / "activity.jsonl").read_text("utf-8").splitlines()
        events = [json.loads(l)["ev"] for l in activity]
        for expected in ("add", "search", "delete", "speaker", "add_refused", "person_add"):
            self.assertIn(expected, events)
        archive = list((self.tmp / "default" / "archive").glob("*.jsonl"))
        self.assertTrue(archive and "Marion" in archive[0].read_text("utf-8"))

if __name__ == "__main__":
    unittest.main()
