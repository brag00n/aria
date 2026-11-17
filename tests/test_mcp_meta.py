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


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class SondageTests(unittest.TestCase):
    """Un serveur MCP distant éteint ou muet ne doit pas retarder la connexion des autres."""

    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_port_ouvert(self):
        import socket
        from components.mcp import server_reachable
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            ok, why = asyncio.run(server_reachable(f"http://127.0.0.1:{srv.getsockname()[1]}/mcp"))
        finally:
            srv.close()
        self.assertEqual((ok, why), (True, ""))

    def test_port_ferme_refus_immediat(self):
        import socket
        import time
        from components.mcp import server_reachable
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        t0 = time.time()
        ok, why = asyncio.run(server_reachable(f"http://127.0.0.1:{port}/mcp"))
        self.assertFalse(ok)
        self.assertIn("Error", why)
        self.assertLess(time.time() - t0, 3)

    def test_hote_muet_borne_par_le_delai(self):
        import time
        from components import mcp as mcp_mod
        original = asyncio.open_connection

        async def muet(*a, **k):
            await asyncio.sleep(30)
        asyncio.open_connection = muet
        try:
            t0 = time.time()
            ok, why = asyncio.run(mcp_mod.server_reachable("http://10.255.255.1:8790/mcp", timeout=0.3))
        finally:
            asyncio.open_connection = original
        self.assertFalse(ok)
        self.assertIn("pas de réponse", why)
        self.assertLess(time.time() - t0, 3)

    def test_url_invalide(self):
        from components.mcp import server_reachable
        for url in ("", "pas une url", "http://"):
            self.assertFalse(asyncio.run(server_reachable(url))[0], url)

    def test_un_serveur_injoignable_est_ignore_et_les_autres_se_connectent(self):
        """_connect_all passe au serveur suivant au lieu de rester bloqué sur le robot éteint."""
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        m = Mcp({"servers": {"robot": {"url": f"http://127.0.0.1:{port}/mcp"}}})
        asyncio.run(m._connect_all())
        self.assertEqual(m.tool_registry, {})


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class CrochetTests(unittest.TestCase):
    """`Mcp.on_tool_done` : appelé après un outil réussi ; une panne du crochet ne fait jamais échouer l'outil."""

    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def mcp(self, reponse="OK"):
        m = Mcp({"servers": {}})

        class S:
            async def call_tool(self, name, arguments=None, meta=None, **kw):
                return Resultat(reponse)
        m.tool_registry["memoriser"] = S()
        return m

    def test_appele_apres_un_outil_reussi_avec_son_nom(self):
        m, vus = self.mcp(), []

        async def crochet(nom):
            vus.append(nom)
        m.on_tool_done = crochet
        self.assertEqual(asyncio.run(m.call_tool("memoriser", {})), "OK")
        self.assertEqual(vus, ["memoriser"])

    def test_une_panne_du_crochet_ne_change_pas_la_reponse(self):
        m = self.mcp("OK m_1")

        async def crochet(nom):
            raise RuntimeError("boum")
        m.on_tool_done = crochet
        self.assertEqual(asyncio.run(m.call_tool("memoriser", {})), "OK m_1")

    def test_aucun_crochet_par_defaut_et_pas_pour_un_outil_inconnu(self):
        m, vus = self.mcp(), []
        self.assertIsNone(m.on_tool_done)

        async def crochet(nom):
            vus.append(nom)
        m.on_tool_done = crochet
        asyncio.run(m.call_tool("inconnu", {}))
        self.assertEqual(vus, [])

    def test_les_outils_ui_ne_sont_pas_notes_comme_outils_du_llm(self):
        from components import echanges
        m = self.mcp()
        m.tool_registry["ui_interlocuteur"] = m.tool_registry["memoriser"]

        async def tour():
            e = echanges.start("aria", "x", "c")
            await m.call_tool("ui_interlocuteur", {})
            await m.call_tool("memoriser", {})
            return e.outils
        self.assertEqual(asyncio.run(tour()), ["memoriser"])


if __name__ == "__main__":
    unittest.main()
