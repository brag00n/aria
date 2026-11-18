"""Agent par défaut « aria » : moteur et MCP PARTAGÉS avec l'interface, jamais fermés par la session ; registre et clés existent même sans `agents.enabled`.
Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import asyncio
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from starlette.testclient import TestClient
    from components.agents import api, keys, profile, server, sessions
    HAVE = True
except ImportError:
    HAVE = False

ORIGIN = "https://app.example.com"
CONFIG = {
    "active_modules": {"llm": "Llm_Anthropic"},
    "llm": {"Llm_Anthropic": {"params": {"backend_type": "anthropic", "system_message": "Tu es Aria."}}},
    "mcp": {"params": {"servers": {"web": {"url": "http://x/mcp"}, "prive": {"url": "http://p/mcp"}}}},
    "conversation": {"enabled": True, "open_phrases": ["bonjour aria"], "close_phrases": ["merci aria"], "stop_phrases": ["stop"]},
    "agents": {"private_servers": ["prive"], "profiles": {"normal": {"servers": ["web"]}}},
}


class MoteurGlobal:
    """Faux Llm global d'Aria : garde une trace de ses appels et de ses fermetures."""

    def __init__(self):
        self.user_aware_messages = {}
        self.calls = []

    @property
    def engine(self):
        return self

    async def get_answer_web(self, tts, query, user, contexte=""):
        self.calls.append((query, user, contexte))
        self.user_aware_messages.setdefault(user, []).append(query)
        yield "Réponse d'Aria", None, None


class McpGlobal:
    def __init__(self):
        self.connexions = 0
        self.fermetures = 0
        self.on_tool_done = None

    async def ensure_connected(self):
        self.connexions += 1

    async def close(self):
        self.fermetures += 1

    def has_tool(self, nom):
        return False


class MoteurAgent:
    def __init__(self, nom):
        self.nom = nom
        self.user_aware_messages = {}

    async def get_answer_web(self, tts, query, user, contexte=""):
        yield f"{self.nom}:{query}", None, None


class McpAgent(McpGlobal):
    pass


@unittest.skipUnless(HAVE, "starlette absent (lancer dans le conteneur)")
class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aria_agent_"))
        self.config = copy.deepcopy(CONFIG)
        self.keys = keys.KeyStore(self.tmp / "data")
        self.global_llm, self.global_mcp = MoteurGlobal(), McpGlobal()
        self.construits = []

        def factory(prof, identity=None):
            e, m = MoteurAgent(prof["name"]), McpAgent()
            self.construits.append((prof["name"], m))
            return e, m
        self.registre = sessions.SessionRegistry(profile.ConfigProfiles(lambda: self.config), factory, max_sessions=5, ttl_s=900,
                                                 shared_engine=lambda: (self.global_llm, self.global_mcp))
        self.deps = api.Deps(keys=self.keys, sessions=self.registre, stt=None, tts=None, decode_audio=None, on_audit=lambda *a, **k: None, static_dir=str(self.tmp))
        self.client = TestClient(api.create_app(self.deps))
        self.client.__enter__()
        self.cle_aria = self.keys.system_key
        self.vue, self.cle = self.keys.create("Robot", [ORIGIN], profile="normal", limits={"rate_per_min": 100, "daily_turns": 100})

    def tearDown(self):
        self.client.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hdr(self, cle, origin=None):
        h = {"Authorization": f"Bearer {cle}"}
        if origin:
            h["Origin"] = origin
        return h

    def turn(self, texte, session="s1", cle=None, **kw):
        return self.client.post("/v1/turn", json={"text": texte, "session": session, **kw}, headers=self.hdr(cle or self.cle_aria))

    @staticmethod
    def events(resp):
        out = []
        for bloc in resp.text.split("\n\n"):
            if bloc.strip():
                nom = next(l[7:] for l in bloc.splitlines() if l.startswith("event: "))
                out.append((nom, json.loads(next(l[6:] for l in bloc.splitlines() if l.startswith("data: ")))))
        return out


class MoteurPartageTests(Base):
    def test_l_agent_aria_utilise_le_moteur_global_pas_un_moteur_a_lui(self):
        ev = self.events(self.turn("quelle heure"))
        self.assertEqual(ev[-1][0], "done")
        self.assertEqual([c[0] for c in self.global_llm.calls], ["quelle heure"])
        self.assertEqual(self.construits, [])                                        # aucun Llm ni Mcp construit pour lui

    def test_un_autre_agent_garde_son_propre_moteur(self):
        self.events(self.turn("a", cle=self.cle))
        self.assertEqual([n for n, _ in self.construits], ["normal"])
        self.assertEqual(self.global_llm.calls, [])

    def test_l_historique_de_l_agent_aria_est_celui_de_sa_session_dans_le_moteur_global(self):
        self.turn("un", session="a")
        self.turn("deux", session="b")
        self.assertEqual(sorted(self.global_llm.user_aware_messages), [f"aria:a", f"aria:b"])

    def test_la_session_ne_connecte_ni_ne_ferme_le_mcp_global(self):
        self.turn("a")
        self.assertEqual((self.global_mcp.connexions, self.global_mcp.fermetures), (0, 0))
        asyncio.get_event_loop_policy()
        self.client.post("/v1/close", json={"session": "s1"}, headers=self.hdr(self.cle_aria))
        self.client.post("/v1/reset", json={"session": "s1"}, headers=self.hdr(self.cle_aria))
        self.assertEqual(self.global_mcp.fermetures, 0)

    def test_un_agent_ordinaire_ferme_bien_son_propre_mcp(self):
        self.turn("a", cle=self.cle)
        self.client.post("/v1/close", json={"session": "s1"}, headers=self.hdr(self.cle))
        (_, mcp), = self.construits
        self.assertEqual((mcp.connexions, mcp.fermetures), (1, 1))

    def test_le_flag_de_possession(self):
        self.turn("a")
        self.turn("b", cle=self.cle)
        par_cle = {s.key_id: s.owned for s in self.registre.sessions.values()}
        self.assertEqual(par_cle, {"aria": False, self.vue["id"]: True})

    def test_sans_moteur_partage_l_agent_aria_construit_le_sien(self):
        self.registre.shared_engine = None
        self.events(self.turn("a"))
        self.assertEqual([n for n, _ in self.construits], ["aria"])
        self.assertTrue(all(s.owned for s in self.registre.sessions.values()))

    def test_la_surveillance_de_l_interlocuteur_ne_touche_jamais_le_mcp_partage(self):
        """`publish_speaker` est vrai pour l'agent système, mais `on_tool_done` du MCP global ne doit PAS recevoir de crochet de session."""
        self.turn("a")
        self.assertIsNone(self.global_mcp.on_tool_done)
        s = next(s for s in self.registre.sessions.values() if s.key_id == "aria")
        self.assertFalse(s.watch_speaker)


class ProfilParDefautTests(Base):
    def test_le_profil_aria_existe_sans_etre_ecrit_dans_la_config(self):
        self.assertNotIn("aria", self.config["agents"]["profiles"])
        self.assertEqual(self.turn("a").status_code, 200)

    def test_un_profil_aria_ecrit_par_l_administrateur_l_emporte(self):
        self.config["agents"]["profiles"]["aria"] = {"description": "Personnalisé", "servers": ["web"]}
        prof = self.registre._load_profile(self.keys.record("aria"))
        self.assertEqual(prof["description"], "Personnalisé")

    def test_il_suit_la_conversation_de_la_config_d_aria(self):
        prof = self.registre._load_profile(self.keys.record("aria"))
        self.assertEqual(prof["conversation"]["open_phrases"], ["bonjour aria"])
        self.config["conversation"]["open_phrases"] = ["salut"]
        self.assertEqual(self.registre._load_profile(self.keys.record("aria"))["conversation"]["open_phrases"], ["salut"])

    def test_il_donne_tous_les_serveurs_prives_compris(self):
        prof = self.registre._load_profile(self.keys.record("aria"))
        self.assertEqual(sorted(prof["mcp"]["servers"]), ["prive", "web"])

    def test_le_profil_aria_n_est_pas_cree_pour_un_autre_agent(self):
        autre, _ = self.keys.create("Autre", [ORIGIN], profile="aria", limits={"rate_per_min": 100, "daily_turns": 100})
        prof = self.registre._load_profile(self.keys.record(autre["id"]))
        self.assertEqual(prof["mcp"]["servers"], {})                                 # même profil implicite, mais SANS le droit « tous les serveurs »
        self.assertNotIn("aria", profile.ConfigProfiles(lambda: self.config).names())

    def test_les_mots_cles_de_la_conversation_s_appliquent_a_l_agent_aria(self):
        ev = self.events(self.turn("stop"))
        self.assertEqual(ev[-1], ("done", {"stopped": True, "reason": "keyword"}))
        self.assertEqual(self.global_llm.calls, [])


class PontEntreBouclesTests(unittest.TestCase):
    """Deux vraies boucles : celle de l'API (le test) et celle de l'interface (un thread). L'agent `aria` s'exécute DANS celle de l'interface, les autres dans celle de l'API."""

    def setUp(self):
        import threading
        from components.agents import bridge
        self.tmp = Path(tempfile.mkdtemp(prefix="aria_pont_"))
        self.config = copy.deepcopy(CONFIG)
        self.keys = keys.KeyStore(self.tmp / "data")
        self.interface = asyncio.new_event_loop()
        self.fil = threading.Thread(target=lambda: (asyncio.set_event_loop(self.interface), self.interface.run_forever()), daemon=True, name="boucle-interface")
        self.fil.start()
        self.holder = bridge.LoopHolder()
        self.holder.set(self.interface)
        self.fils = {}

        class Llm:
            def __init__(inner, nom):
                inner.nom, inner.user_aware_messages = nom, {}

            @property
            def engine(inner):
                return inner

            async def get_answer_web(inner, tts, query, user, contexte=""):
                import threading as t
                self.fils[inner.nom] = t.current_thread().name
                yield "ok", None, None
        self.Llm = Llm
        self.registre = sessions.SessionRegistry(profile.ConfigProfiles(lambda: self.config), lambda prof, identity=None: (Llm("agent"), McpAgent()),
                                                 max_sessions=5, ttl_s=900, shared_engine=lambda: (Llm("aria"), McpGlobal()))
        self.deps = api.Deps(keys=self.keys, sessions=self.registre, on_audit=lambda *a, **k: None, static_dir=str(self.tmp), loop_holder=self.holder)
        self.client = TestClient(api.create_app(self.deps))
        self.client.__enter__()
        self.vue, self.cle = self.keys.create("Robot", [ORIGIN], profile="normal", limits={"rate_per_min": 100, "daily_turns": 100})

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.interface.call_soon_threadsafe(self.interface.stop)
        self.fil.join(2)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def turn(self, cle, session="p"):
        return self.client.post("/v1/turn", json={"text": "bonjour", "session": session}, headers={"Authorization": f"Bearer {cle}"})

    def test_l_agent_aria_s_execute_dans_la_boucle_de_l_interface(self):
        r = self.turn(self.keys.system_key)
        self.assertIn("event: done", r.text)
        self.assertEqual(self.fils["aria"], "boucle-interface")

    def test_un_autre_agent_reste_dans_la_boucle_de_l_api(self):
        r = self.turn(self.cle)
        self.assertIn("event: done", r.text)
        self.assertNotEqual(self.fils["agent"], "boucle-interface")

    def test_sans_boucle_d_interface_connue_l_agent_aria_fonctionne_quand_meme(self):
        from components.agents import bridge
        self.deps.loop_holder = bridge.LoopHolder()
        r = self.turn(self.keys.system_key, session="q")
        self.assertIn("event: done", r.text)
        self.assertNotEqual(self.fils["aria"], "boucle-interface")

    def test_un_agent_systeme_sans_moteur_partage_ne_passe_pas_par_le_pont(self):
        """Son moteur est à lui, ouvert dans la boucle de l'API : le faire passer par la boucle de l'interface le casserait (anyio exige la même tâche)."""
        self.registre.shared_engine = None
        r = self.turn(self.keys.system_key, session="solo")
        self.assertIn("event: done", r.text)
        self.assertNotEqual(self.fils["agent"], "boucle-interface")
        self.assertNotIn("aria", self.fils)

    def test_les_evenements_arrivent_dans_l_ordre_a_travers_le_pont(self):
        ev = [l[7:] for l in self.turn(self.keys.system_key, session="o").text.splitlines() if l.startswith("event: ")]
        self.assertEqual(ev[0], "state")
        self.assertEqual(ev[-1], "done")
        self.assertIn("text", ev)


class SansLimitesTests(Base):
    def test_l_agent_aria_n_est_jamais_limite(self):
        for i in range(12):
            self.assertEqual(self.turn(f"m{i}").status_code, 200, i)

    def test_un_agent_ordinaire_a_ses_limites(self):
        v, c = self.keys.create("Limite", [ORIGIN], profile="normal", limits={"rate_per_min": 2, "daily_turns": 2})
        codes = [self.turn(f"m{i}", cle=c).status_code for i in range(4)]
        self.assertIn(429, codes)


class ServeurTests(unittest.TestCase):
    """`server.start` assemble TOUJOURS les agents ; il n'ouvre le port que si `agents.enabled`."""

    @unittest.skipUnless(HAVE, "starlette absent")
    def test_sans_enabled_les_agents_existent_mais_rien_n_est_ouvert(self):
        import unittest.mock as mock
        tmp = Path(tempfile.mkdtemp(prefix="aria_srv_"))
        try:
            cfg = copy.deepcopy(CONFIG)
            cfg["agents"]["data_dir"] = str(tmp / "d").replace("\\", "/")
            with mock.patch.object(server.utils, "aria_path", lambda *p: str(tmp.joinpath(*p))):
                with mock.patch.object(server.compat, "data_dir", lambda root, conf: tmp / "d"):
                    ag = server.start(cfg, None, None, {}, shared_engine=lambda: (MoteurGlobal(), McpGlobal()))
            self.assertIsNone(ag.server)
            self.assertIsNone(ag.admin)
            self.assertIsNotNone(ag.agents_deps.keys.get("aria"))
            self.assertEqual(ag.agents_deps.sessions.shared_engine is not None, True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_start_et_build_transmettent_la_boucle_de_l_interface_aux_dependances(self):
        import unittest.mock as mock
        from components.agents import bridge
        tmp = Path(tempfile.mkdtemp(prefix="aria_srv2_"))
        try:
            holder = bridge.LoopHolder()
            with mock.patch.object(server.compat, "data_dir", lambda root, conf: tmp / "d"):
                app, deps, store, profils = server.build(copy.deepcopy(CONFIG), None, None, None, shared_engine=lambda: (None, None), loop_holder=holder)
            self.assertIs(deps.loop_holder, holder)
            self.assertIsNotNone(deps.sessions.shared_engine)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(HAVE, "starlette absent")
    def test_agents_deps_est_le_nom_lu_par_le_reste_d_aria(self):
        ag = server.Agents("D", "S", "P", "A")
        self.assertEqual(ag.agents_deps, "D")


if __name__ == "__main__":
    unittest.main()
