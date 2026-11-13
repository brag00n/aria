"""API tierce (Starlette) avec LLM, STT et TTS factices. Nécessite starlette/httpx : ignoré si absents (ils sont dans le conteneur d'Aria)."""
import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from starlette.testclient import TestClient
    HAVE_STARLETTE = True
except ImportError:
    HAVE_STARLETTE = False

from components.access import api, keys, profile, sessions  # noqa: E402

ORIGIN = "https://app.example.com"


class FakeEngine:
    """Moteur LLM factice : mêmes signatures que LlmConnectAnthropic (get_answer_web asynchrone, historique par utilisateur)."""

    def __init__(self, name="a", slow=False):
        self.name, self.slow = name, slow
        self.user_aware_messages = {}
        self.calls = []

    async def get_answer_web(self, tts, query, user, contexte=""):
        self.calls.append((query, user))
        self.user_aware_messages.setdefault(user, []).append(query)
        yield "Bonjour", None, None
        if self.slow:
            await asyncio.sleep(0.3)
        yield "Bonjour" + "\n\n> 🛠️ Outil utilisé : `meteo`\n\n", None, None
        wav = None
        if tts is not None:
            wav = tts.run_tts_to_file("Bonjour", user_id=user)
        yield "Bonjour" + "\n\n> 🛠️ Outil utilisé : `meteo`\n\n" + f" {self.name}:{query}", wav, "Bonjour"


class FakeLlm:
    def __init__(self, engine):
        self.engine = engine

    async def get_answer_web(self, tts, query, user, contexte=""):
        async for v in self.engine.get_answer_web(tts, query, user, contexte):
            yield v


class FakeTts:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.texts = []

    def run_tts_to_file(self, text, user_id="x"):
        self.texts.append(text)
        p = self.dir / f"t_{len(self.texts)}.wav"
        p.write_bytes(b"RIFFfakewav")
        return str(p)


class FakeStt:
    def __init__(self):
        self.calls = 0

    def transcribe_translate(self, samples):
        self.calls += 1
        return "quelle heure est-il"


def _decode(raw):
    import numpy as np  # noqa: F401  (absent de l'hôte : le test est alors ignoré)
    return np.zeros(16000, dtype="float32")


@unittest.skipUnless(HAVE_STARLETTE, "starlette absent (lancer dans le conteneur)")
class ApiCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="accapi_"))
        (self.tmp / "profils").mkdir()
        (self.tmp / "profils" / "portail.json").write_text(json.dumps({"description": "Portail", "mcp": {"servers": {}}}), "utf-8")
        (self.tmp / "profils" / "autre.json").write_text(json.dumps({"description": "Autre", "mcp": {"servers": {}}}), "utf-8")
        self.keys = keys.KeyStore(self.tmp / "data")
        self.engines = []

        def factory(prof):
            e = FakeEngine(prof["name"], slow=getattr(self, "slow", False))
            self.engines.append(e)
            return FakeLlm(e), None
        self.sessions = sessions.SessionRegistry(profile.Profiles(self.tmp / "profils"), factory, max_sessions=3, ttl_s=900)
        self.tts = FakeTts(self.tmp)
        self.stt = FakeStt()
        self.audit = []
        self.deps = api.Deps(keys=self.keys, sessions=self.sessions, stt=self.stt, tts=self.tts, decode_audio=_decode,
                             on_audit=lambda ev, **kw: self.audit.append((ev, kw)))
        self.client = TestClient(api.create_app(self.deps))
        self.view, self.key = self.keys.create("Portail", [ORIGIN], profile="portail", limits={"rate_per_min": 100, "daily_turns": 100})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def hdr(self, key=None, origin=ORIGIN):
        h = {"Authorization": f"Bearer {key or self.key}"}
        if origin:
            h["Origin"] = origin
        return h

    @staticmethod
    def events(resp):
        """Événements SSE d'une réponse : liste de (nom, données)."""
        out = []
        for bloc in resp.text.split("\n\n"):
            if not bloc.strip():
                continue
            nom = next(l[7:] for l in bloc.splitlines() if l.startswith("event: "))
            data = json.loads(next(l[6:] for l in bloc.splitlines() if l.startswith("data: ")))
            out.append((nom, data))
        return out

    def turn(self, body=None, **kw):
        return self.client.post("/v1/turn", json=body if body is not None else {"text": "bonjour"}, headers=kw.pop("headers", None) or self.hdr())


class AuthTests(ApiCase):
    def test_sans_cle_401(self):
        r = self.client.post("/v1/turn", json={"text": "x"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "unauthorized")

    def test_cle_invalide_revoquee_expiree_meme_message(self):
        a = self.client.post("/v1/turn", json={"text": "x"}, headers=self.hdr("ak_" + "z" * 43))
        self.keys.revoke(self.view["id"])
        b = self.turn()
        self.assertEqual((a.status_code, b.status_code), (401, 401))
        self.assertEqual(a.json(), b.json())                                           # rien à apprendre en devinant des clés

    def test_origine_refusee_403_sans_cors(self):
        r = self.turn(headers=self.hdr(origin="https://evil.example.com"))
        self.assertEqual(r.status_code, 403)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_origine_autorisee_recoit_cors_exact(self):
        r = self.turn()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["access-control-allow-origin"], ORIGIN)
        self.assertEqual(r.headers["vary"], "Origin")

    def test_client_sans_origine_accepte_sans_cors(self):
        r = self.turn(headers=self.hdr(origin=None))
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_x_api_key(self):
        r = self.client.post("/v1/turn", json={"text": "x"}, headers={"X-Api-Key": self.key})
        self.assertEqual(r.status_code, 200)

    def test_preflight(self):
        r = self.client.options("/v1/turn", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r.headers["access-control-allow-origin"], ORIGIN)
        r = self.client.options("/v1/turn", headers={"Origin": "https://evil.example.com"})
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_blocage_apres_echecs_repetes(self):
        for _ in range(8):
            self.client.post("/v1/turn", json={"text": "x"}, headers=self.hdr("ak_" + "z" * 43))
        r = self.turn()                                                                # même IP, bonne clé : bloquée
        self.assertEqual(r.status_code, 429)
        self.assertIn("retry-after", r.headers)
        self.assertTrue(any(ev == "blocked" for ev, _ in self.audit))

    def test_la_cle_n_est_jamais_dans_l_audit(self):
        self.client.post("/v1/turn", json={"text": "x"}, headers=self.hdr("ak_" + "z" * 43))
        self.turn()
        self.assertNotIn("zzzz", json.dumps(self.audit))
        self.assertNotIn(self.key, json.dumps(self.audit))


class LimitesTests(ApiCase):
    def test_debit_par_cle(self):
        v, k = self.keys.create("Lent", [ORIGIN], profile="portail", limits={"rate_per_min": 2, "daily_turns": 100})
        codes = [self.turn(headers=self.hdr(k)).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_quota_journalier(self):
        v, k = self.keys.create("Quota", [ORIGIN], profile="portail", limits={"rate_per_min": 100, "daily_turns": 2})
        codes = [self.turn(headers=self.hdr(k)).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])
        self.assertEqual(self.client.post("/v1/turn", json={"text": "x"}, headers=self.hdr(k)).json()["error"]["code"], "quota")

    def test_texte_trop_long_vide_et_corps_invalides(self):
        self.assertEqual(self.turn({"text": "x" * 2001}).status_code, 413)
        self.assertEqual(self.turn({"text": "   "}).status_code, 400)
        self.assertEqual(self.turn({}).status_code, 400)
        self.assertEqual(self.turn({"text": "a", "audio": "data:audio/wav;base64,AAAA"}).status_code, 400)
        r = self.client.post("/v1/turn", content=b"{pas json", headers={**self.hdr(), "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/v1/turn", content=b"x", headers={**self.hdr(), "Content-Type": "text/plain"})
        self.assertEqual(r.status_code, 415)
        r = self.client.post("/v1/turn", json=[1, 2], headers=self.hdr())
        self.assertEqual(r.status_code, 400)

    def test_corps_trop_gros(self):
        r = self.client.post("/v1/turn", content=b"{" + b" " * (9 * 1024 * 1024) + b"}", headers={**self.hdr(), "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)


class TourTexteTests(ApiCase):
    def test_flux_d_evenements(self):
        ev = self.events(self.turn({"text": "salut", "speak": False}))
        noms = [n for n, _ in ev]
        self.assertEqual(noms[0], "state")
        self.assertIn("text", noms)
        self.assertEqual(noms[-1], "done")
        self.assertNotIn("audio", noms)
        self.assertEqual([d["name"] for n, d in ev if n == "tool"], ["meteo"])        # l'outil est annoncé une seule fois
        dernier = [d["text"] for n, d in ev if n == "text"][-1]
        self.assertIn("a:salut", dernier.replace("portail", "a"))
        self.assertNotIn("🛠️", dernier)                                                # la ligne d'outil est retirée du texte

    def test_la_voix_renvoie_des_blocs_audio_et_supprime_les_fichiers(self):
        ev = self.events(self.turn({"text": "salut"}))
        audio = [d for n, d in ev if n == "audio"]
        self.assertEqual(len(audio), 1)
        self.assertEqual(base64.b64decode(audio[0]["data"]), b"RIFFfakewav")
        self.assertEqual(audio[0]["format"], "wav")
        self.assertEqual([p for p in self.tmp.glob("t_*.wav")], [])

    def test_voix_coupee_pour_la_cle(self):
        v, k = self.keys.create("Muet", [ORIGIN], profile="portail", voice=False)
        ev = self.events(self.turn({"text": "salut"}, headers=self.hdr(k)))
        self.assertNotIn("audio", [n for n, _ in ev])
        self.assertEqual(self.tts.texts, [])

    def test_erreur_du_llm_devient_une_erreur_structuree(self):
        async def boom(tts, query, user, contexte=""):
            yield "Erreur LLM: clé secrète sk-abc rejetée", None, None
        self.deps.sessions.factory = lambda prof: (type("L", (), {"get_answer_web": staticmethod(boom)}), None)
        ev = self.events(self.turn())
        err = [d for n, d in ev if n == "error"]
        self.assertEqual(err[0]["code"], "llm")
        self.assertNotIn("sk-abc", json.dumps(ev))                                     # le détail du fournisseur n'est pas relayé


class TourVoixTests(ApiCase):
    AUDIO = "data:audio/wav;base64," + base64.b64encode(b"RIFFxxxx").decode()

    def test_tour_vocal(self):
        ev = self.events(self.turn({"audio": self.AUDIO, "speak": False}))
        self.assertEqual([d["text"] for n, d in ev if n == "user_text"], ["quelle heure est-il"])
        self.assertEqual(self.stt.calls, 1)
        self.assertEqual(self.engines[0].calls[0][0], "quelle heure est-il")

    def test_voix_interdite_a_la_cle(self):
        v, k = self.keys.create("Texte", [ORIGIN], profile="portail", voice=False)
        r = self.turn({"audio": self.AUDIO}, headers=self.hdr(k))
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (403, "voice_disabled"))

    def test_audio_mal_forme(self):
        self.assertEqual(self.turn({"audio": "pas une data uri"}).status_code, 400)
        self.assertEqual(self.turn({"audio": 12}).status_code, 400)

    def test_audio_trop_long(self):
        self.deps.max_audio_s = 0.5
        ev = self.events(self.turn({"audio": self.AUDIO}))
        self.assertEqual([d["code"] for n, d in ev if n == "error"], ["audio_too_long"])

    def test_silence_ne_lance_pas_le_llm(self):
        self.stt.transcribe_translate = lambda s: "  "
        ev = self.events(self.turn({"audio": self.AUDIO}))
        self.assertEqual(ev[-1][0], "done")
        self.assertEqual(self.engines[0].calls, [])


class SessionsTests(ApiCase):
    def test_historique_isole_par_session_et_par_cle(self):
        self.turn({"text": "un", "session": "s1"})
        self.turn({"text": "deux", "session": "s2"})
        v2, k2 = self.keys.create("Autre appli", [ORIGIN], profile="autre")
        self.turn({"text": "trois", "session": "s1"}, headers=self.hdr(k2))
        users = [u for e in self.engines for u in e.user_aware_messages]
        self.assertEqual(len(set(users)), 3)
        self.assertEqual(len(self.engines), 3)                                          # un moteur par (clé, session)
        self.assertEqual(len(self.sessions.sessions), 3)

    def test_meme_session_reutilise_le_moteur_et_garde_l_historique(self):
        self.turn({"text": "un", "session": "s1"})
        self.turn({"text": "deux", "session": "s1"})
        self.assertEqual(len(self.engines), 1)
        self.assertEqual(list(self.engines[0].user_aware_messages.values()), [["un", "deux"]])

    def test_reset_efface_l_historique(self):
        self.turn({"text": "un", "session": "s1"})
        r = self.client.post("/v1/reset", json={"session": "s1"}, headers=self.hdr())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.engines[0].user_aware_messages, {})

    def test_plafond_de_sessions(self):
        for i in range(3):
            self.assertEqual(self.turn({"text": "x", "session": f"s{i}"}).status_code, 200)
        r = self.turn({"text": "x", "session": "s9"})
        self.assertEqual(r.status_code, 503)

    def test_cle_sans_profil_ou_profil_inconnu(self):
        v, k = self.keys.create("Nu", [ORIGIN])
        self.assertEqual(self.turn(headers=self.hdr(k)).status_code, 403)
        v, k = self.keys.create("Fantome", [ORIGIN], profile="inexistant")
        self.assertEqual(self.turn(headers=self.hdr(k)).status_code, 500)

    def test_identifiant_de_session_nettoye(self):
        self.turn({"text": "x", "session": "../../etc/passwd"})
        self.assertTrue(all("/" not in s and ".." not in s for s in self.sessions.sessions))

    def test_stop(self):
        r = self.client.post("/v1/stop", json={"session": "s1"}, headers=self.hdr())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(next(iter(self.sessions.sessions.values())).stop.is_set())

    def test_health(self):
        r = self.client.get("/v1/health", headers=self.hdr())
        self.assertEqual(r.json()["ok"], True)
        self.assertEqual(self.client.get("/v1/health").status_code, 401)


class FichiersPublicsTests(ApiCase):
    def setUp(self):
        super().setUp()
        (self.tmp / "static" / "silero_v6.2").mkdir(parents=True)
        (self.tmp / "static" / "aria-client.js").write_text("window.AriaClient={};", "utf-8")
        (self.tmp / "static" / "silero_v6.2" / "silero_vad.onnx").write_bytes(b"onnx")
        (self.tmp / "static" / "secret.txt").write_text("ne pas servir", "utf-8")
        self.deps.static_dir = str(self.tmp / "static")

    def test_le_script_est_servi_sans_cle_avec_le_bon_type(self):
        r = self.client.get("/aria-client.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("javascript", r.headers["content-type"])
        self.assertEqual(r.text, "window.AriaClient={};")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")

    def test_le_modele_vad_est_servi(self):
        r = self.client.get("/assets/silero_vad.onnx")
        self.assertEqual((r.status_code, r.content), (200, b"onnx"))
        self.assertEqual(r.headers["access-control-allow-origin"], "*")                    # fichiers publics : lecture seule depuis toute page

    def test_liste_fermee_aucun_autre_fichier(self):
        for chemin in ("/assets/secret.txt", "/assets/../secret.txt", "/assets/..%2Fsecret.txt", "/assets/%2e%2e/aria-client.js",
                       "/assets/keys.json", "/assets/", "/static/aria-client.js"):
            self.assertEqual(self.client.get(chemin).status_code, 404, chemin)

    def test_sans_dossier_statique_configure(self):
        self.deps.static_dir = ""
        self.assertEqual(self.client.get("/aria-client.js").status_code, 404)

    def test_les_routes_d_api_restent_protegees(self):
        self.assertEqual(self.client.get("/v1/health").status_code, 401)


class ProfilTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="accprof_"))
        self.p = profile.Profiles(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_noms_invalides_refuses(self):
        for nom in ("../secret", "a/b", "a\\b", "", ".env", "x" * 61, None, "a.json", "..", "a b"):
            self.assertFalse(profile.valid_name(nom), nom)
            with self.assertRaises(profile.ProfileError):
                self.p.load(nom)

    def test_profil_valide_et_apercu_sans_secrets(self):
        (self.tmp / "ok.json").write_text(json.dumps({"description": "d", "mcp": {"servers": {
            "web": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer <Jeton>"}},
            "loc": {"run": "uv", "args": ["run", "a.py"], "env": {"K": "<Secret>"}}}}}), "utf-8")
        d = self.p.describe("ok")
        self.assertEqual({s["name"]: s["type"] for s in d["servers"]}, {"web": "http", "loc": "stdio"})
        self.assertNotIn("Bearer", json.dumps(d))
        self.assertNotIn("Secret", json.dumps(d))

    def test_profils_invalides(self):
        for contenu in ("pas json", "[1]", json.dumps({"mcp": {"servers": []}}), json.dumps({"mcp": {"servers": {"a": 3}}}),
                        json.dumps({"mcp": {"servers": {"a": {}}}})):
            (self.tmp / "bad.json").write_text(contenu, "utf-8")
            with self.assertRaises(profile.ProfileError):
                self.p.load("bad")
        with self.assertRaises(profile.ProfileError):
            self.p.load("absent")

    def test_profil_sans_mcp_n_a_aucun_outil(self):
        (self.tmp / "vide.json").write_text("{}", "utf-8")
        self.assertEqual(self.p.load("vide")["mcp"], {"servers": {}})

    def test_liste(self):
        (self.tmp / "b.json").write_text("{}", "utf-8")
        (self.tmp / "a.json").write_text("{}", "utf-8")
        (self.tmp / "..x.json").write_text("{}", "utf-8")
        self.assertEqual(self.p.names(), ["a", "b"])


class MoteurTests(unittest.TestCase):
    def test_le_llm_local_est_refuse_pour_un_tiers(self):
        cfg = {"active_modules": {"llm": "Llm_Llama_local"}, "llm": {"Llm_Llama_local": {"params": {"backend_type": "local"}}}}
        with self.assertRaises(profile.ProfileError):
            sessions.build_engine({"llm": "", "system_message": "", "mcp": {"servers": {}}}, cfg)
        cfg = {"active_modules": {"llm": "X"}, "llm": {"X": {"params": {}}}}                  # backend absent = local par défaut
        with self.assertRaises(profile.ProfileError):
            sessions.build_engine({"llm": "", "system_message": "", "mcp": {"servers": {}}}, cfg)


if __name__ == "__main__":
    unittest.main()
