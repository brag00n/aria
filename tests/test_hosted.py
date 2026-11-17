"""Serveurs de mémoire hébergés pour des tiers hors foyer (`access.hosted_only`) : validation et non-exposition à l'interface d'Aria."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.agents import profile  # noqa: E402

ARGS = ["run", "/aria/tools/memory/server.py"]
RACINE = Path(__file__).resolve().parent.parent
if not (RACINE / "components" / "mcp.py").exists():                    # copie des tests dans le conteneur : le code est sous /aria
    RACINE = Path("/aria")


def mem(dossier, port=None, ui=None, ns=None):
    env = {"MEMORY_DIR": dossier}
    if port:
        env["MEMORY_HTTP_PORT"] = str(port)
    if ui:
        env["MEMORY_UI_PORT"] = str(ui)
    if ns:
        env["MEMORY_NAMESPACE"] = ns
    return {"run": "uv", "args": ARGS, "env": env}


def conf(servers, hosted=None):
    access = {"enabled": True}
    if hosted is not None:
        access["hosted_only"] = hosted
    return {"mcp": {"params": {"servers": servers}}, "access": access}


FOYER = mem("/aria/models/memory", 7872, ui=7870)
EXTERNE = mem("/aria/models/memoire_acme", 7873)


class ValidationTests(unittest.TestCase):
    def test_configuration_valide(self):
        self.assertEqual(profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": EXTERNE}, ["memoire_acme"])), [])

    def test_sans_hosted_only(self):
        self.assertEqual(profile.validate_hosted(conf({"memoire": FOYER})), [])
        self.assertEqual(profile.hosted_only(conf({"memoire": FOYER})), [])

    def test_serveur_inconnu(self):
        e = profile.validate_hosted(conf({"memoire": FOYER}, ["fantome"]))
        self.assertEqual(len(e), 1)
        self.assertIn("inconnu", e[0])

    def test_sans_canal_http(self):
        e = profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": mem("/aria/models/memoire_acme")}, ["memoire_acme"]))
        self.assertTrue(any("canal HTTP" in x for x in e), e)

    def test_deux_memoires_sur_le_meme_dossier_se_verrouilleraient(self):
        e = profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": mem("/aria/models/memory", 7873)}, ["memoire_acme"]))
        self.assertTrue(any("même dossier" in x for x in e), e)

    def test_meme_dossier_avec_un_autre_espace_de_noms_est_distinct(self):
        self.assertEqual(profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": mem("/aria/models/memory", 7873, ns="acme")}, ["memoire_acme"])), [])

    def test_deux_memoires_sur_le_meme_port(self):
        e = profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": mem("/aria/models/memoire_acme", 7872)}, ["memoire_acme"]))
        self.assertTrue(any("même port" in x for x in e), e)

    def test_une_seule_page_de_gestion(self):
        e = profile.validate_hosted(conf({"memoire": FOYER, "memoire_acme": mem("/aria/models/memoire_acme", 7873, ui=7871)}, ["memoire_acme"]))
        self.assertTrue(any("page de gestion" in x for x in e), e)

    def test_forme_invalide(self):
        for faux in ("memoire", {"a": 1}, [1], None):
            cfg = conf({"memoire": FOYER})
            cfg["access"]["hosted_only"] = faux
            self.assertTrue(profile.validate_hosted(cfg), faux)

    def test_la_liste_ignore_les_elements_invalides(self):
        cfg = conf({"memoire": FOYER})
        cfg["access"]["hosted_only"] = ["a", 3, None, "b"]
        self.assertEqual(profile.hosted_only(cfg), ["a", "b"])


class ProfilTests(unittest.TestCase):
    def test_un_profil_tiers_hors_foyer_reference_son_serveur(self):
        cfg = conf({"memoire": FOYER, "memoire_acme": EXTERNE}, ["memoire_acme"])
        self.assertEqual(profile.validate_profiles({"acme": {"servers": ["memoire_acme"]}}, cfg), [])
        cfg["access"]["profiles"] = {"acme": {"servers": ["memoire_acme"]}}
        p = profile.ConfigProfiles(lambda: cfg).load("acme")
        self.assertEqual(list(p["mcp"]["servers"]), ["memoire_acme"])

    def test_deux_profils_peuvent_partager_le_meme_serveur_hors_foyer(self):
        cfg = conf({"memoire": FOYER, "memoire_acme": EXTERNE}, ["memoire_acme"])
        self.assertEqual(profile.validate_profiles({"a": {"servers": ["memoire_acme"]}, "b": {"servers": ["memoire_acme"]}}, cfg), [])


class InterfaceTests(unittest.TestCase):
    """L'interface d'Aria ne voit jamais les outils d'un serveur hébergé : sinon ils écraseraient ceux de « memoire » (mêmes noms)."""

    def tearDown(self):
        # asyncio.run ferme la boucle par défaut ; d'autres tests (Mcp() la demande à sa construction) en ont besoin d'une
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_les_outils_d_un_serveur_hebergé_ne_sont_pas_enregistres(self):
        try:
            from components.mcp import Mcp
        except ImportError:
            self.skipTest("mcp absent (conteneur)")
        src = (RACINE / "components" / "mcp.py").read_text("utf-8")
        self.assertIn("hosted_only", src)
        async def construire():
            return Mcp({"servers": {}}, hosted_only=["memoire_acme"])
        self.assertEqual(asyncio.run(construire()).hosted_only, frozenset({"memoire_acme"}))

    def test_le_gestionnaire_n_enregistre_pas_les_outils_d_un_serveur_heberge(self):
        """Comportement réel : faux serveurs MCP, le serveur hébergé est démarré mais ses outils restent hors de l'interface."""
        try:
            import components.mcp as mcp_mod
        except ImportError:
            self.skipTest("mcp absent (conteneur)")

        class Outil:
            def __init__(self, nom):
                self.name, self.description, self.inputSchema = nom, "d", {"type": "object", "properties": {}}

        class FauxSession:
            def __init__(self, noms):
                self.noms = noms

            async def initialize(self):
                pass

            async def list_tools(self):
                return type("R", (), {"tools": [Outil(n) for n in self.noms]})()

        class Pile:
            def __init__(self):
                self.entrees = []

            async def enter_async_context(self, ctx):
                self.entrees.append(ctx)
                return ctx

            async def aclose(self):
                pass

        sessions_creees = {}
        original = (mcp_mod.stdio_client, mcp_mod.ClientSession)

        def faux_stdio(params):
            return (f"flux-{params.env.get('NOM')}", "ecriture")

        def faux_session(read, write):
            nom = read.split("-", 1)[1]
            s = FauxSession(["memoriser", "rechercher_souvenir", "ui_etat"] if nom == "foyer" else ["memoriser", "rechercher_souvenir", "ui_etat"])
            sessions_creees[nom] = s
            return s

        mcp_mod.stdio_client, mcp_mod.ClientSession = faux_stdio, faux_session
        try:
            serveurs = {"foyer": {"run": "x", "env": {"NOM": "foyer"}}, "acme": {"run": "x", "env": {"NOM": "acme"}}}
            async def connecter(**kw):
                m = mcp_mod.Mcp({"servers": serveurs}, **kw)               # construit dans une boucle, comme Aria
                m.stack = Pile()
                await m._connect_all()
                return m
            m = asyncio.run(connecter(hosted_only=["acme"]))
            self.assertEqual(sorted(sessions_creees), ["acme", "foyer"])                      # les deux serveurs sont bien démarrés
            self.assertEqual(sorted(t["function"]["name"] for t in m.openai_tools), ["memoriser", "rechercher_souvenir"])        # une seule fois
            self.assertIs(m.tool_registry["memoriser"], sessions_creees["foyer"])               # les outils appartiennent au foyer, pas à acme
            self.assertIs(m.tool_registry["ui_etat"], sessions_creees["foyer"])
            # sans hosted_only, le second écrase le premier : c'est exactement ce que la liste évite
            m2 = asyncio.run(connecter())
            self.assertIs(m2.tool_registry["memoriser"], sessions_creees["acme"])
        finally:
            mcp_mod.stdio_client, mcp_mod.ClientSession = original

    def test_le_mcp_d_aria_recoit_la_liste(self):
        for f in ("llm_connect_anthropic.py", "llm_connect_openai.py"):
            src = (RACINE / "components" / f).read_text("utf-8")
            self.assertIn("hosted_only", src, f)


if __name__ == "__main__":
    unittest.main()
