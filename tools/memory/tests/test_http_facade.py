"""Façade HTTP des accès tiers : authentification par secret, identité imposée par le serveur, outils servis. Nécessite le paquet `mcp` et uvicorn
(dans le conteneur d'Aria) : ignoré sinon. Les contrôles d'identité sont aussi couverts de bout en bout (processus réel) par le script du plan."""
import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
    from starlette.testclient import TestClient
    import http_facade
    HAVE = True
except ImportError:
    HAVE = False

import store as store_mod  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402

SECRET = "secret-de-test-assez-long"


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        self.t += 1
        return self.t


def rpc(method, params=None, rid=1):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


@unittest.skipUnless(HAVE, "mcp / starlette absents (lancer dans le conteneur)")
class FacadeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="httpfacade_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.i = Interlocuteurs(self.s)
        self.svc = Service(self.s, self.i)
        self.i.identifier("Olivier", "Cousinier")
        self.olivier = self.s.active
        self.s.add("Olivier aime la science-fiction", "preference")
        self.app = http_facade.make_app(self.s, self.svc, SECRET)
        self.client = TestClient(self.app.inner, base_url="http://127.0.0.1:7872")        # le transport refuse tout autre en-tête Host
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tool(self, name, args, principal="k_robot", nom="Robot"):
        """Appel d'outil par le transport HTTP, sans passer par le garde d'authentification (testé à part)."""
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if principal:
            headers["X-Memory-Principal"] = principal
        if nom is not None:
            headers["X-Memory-Name"] = nom
        r = self.client.post("/mcp", json=rpc("tools/call", {"name": name, "arguments": args}), headers=headers)
        body = r.json()
        return body.get("result", {}).get("content", [{}])[0].get("text", "") if "result" in body else json.dumps(body)


class IdentiteTests(FacadeCase):
    def test_l_identite_vient_de_l_en_tete_et_cree_la_personne_systeme(self):
        self.assertTrue(self.tool("memoriser", {"fait": "Batterie à 80 pour cent"}).startswith("OK"))
        sys_ = [p for p in self.s.personnes.items if p.get("type") == "systeme"]
        self.assertEqual([(p["nom"], p["cle"]) for p in sys_], [("Robot", "k_robot")])
        mine = [r for r in self.s.records.values() if "Batterie" in r["txt"]]
        self.assertEqual([r["o"] for r in mine], [sys_[0]["id"]])

    def test_sans_identite_rien_n_est_servi(self):
        for principal, nom in ((None, "Robot"), ("k_robot", None), ("", "Robot"), ("k_robot", " ")):
            out = self.tool("memoriser", {"fait": "Tentative"}, principal=principal, nom=nom)
            self.assertIn("Refusé", out)
        self.assertEqual([r for r in self.s.records.values() if "Tentative" in r["txt"]], [])

    def test_le_llm_ne_peut_pas_choisir_son_identite_par_les_arguments(self):
        out = self.tool("memoriser", {"fait": "Je triche", "owner": self.olivier, "principal": self.olivier, "o": self.olivier})
        self.assertTrue(out.startswith("OK") or "error" in out.lower() or "validation" in out.lower())
        for r in self.s.records.values():
            if "Je triche" in r["txt"]:
                self.assertNotEqual(r["o"], self.olivier)

    def test_ne_modifie_jamais_l_actif_d_aria(self):
        self.tool("memoriser", {"fait": "Mesure"})
        self.tool("rechercher_souvenir", {"question": "Mesure"})
        self.tool("relier", {"personne_a": "moi", "lien": "ami", "personne_b": "Olivier"})
        self.assertEqual(self.s.active, self.olivier)

    def test_deux_identites_deux_memoires(self):
        self.tool("memoriser", {"fait": "Secret du robot"}, "k_robot", "Robot")
        self.assertNotIn("Secret du robot", self.tool("rechercher_souvenir", {"question": "secret robot"}, "k_cam", http_facade.encode_name("Caméra")))
        self.assertIn("Secret du robot", self.tool("rechercher_souvenir", {"question": "secret robot"}, "k_robot", "Robot"))

    def test_renommer_l_acces_garde_la_meme_personne_et_les_memes_souvenirs(self):
        self.tool("memoriser", {"fait": "Souvenir durable"}, "k_robot", "Robot")
        self.assertIn("Souvenir durable", self.tool("rechercher_souvenir", {"question": "souvenir durable"}, "k_robot", "Bamboo%204WD"))
        self.assertEqual([p["nom"] for p in self.s.personnes.items if p.get("cle") == "k_robot"], ["Bamboo 4WD"])        # nom décodé

    def test_nom_avec_accents_encode(self):
        self.tool("memoriser", {"fait": "Image enregistrée"}, "k_cam", http_facade.encode_name("Caméra du garage"))
        self.assertEqual([p["nom"] for p in self.s.personnes.items if p.get("cle") == "k_cam"], ["Caméra du garage"])

    def test_cle_interne_et_nom_bornes(self):
        self.tool("memoriser", {"fait": "x"}, "k" * 500, "N" * 500)
        p = [p for p in self.s.personnes.items if p.get("type") == "systeme"][0]
        self.assertLessEqual(len(p["cle"]), 40)
        self.assertLessEqual(len(p["nom"]), 60)


class OutilsTests(FacadeCase):
    def noms(self):
        r = self.client.post("/mcp", json=rpc("tools/list"), headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"})
        return sorted(t["name"] for t in r.json()["result"]["tools"])

    def test_outils_servis_a_un_tiers(self):
        self.assertEqual(self.noms(), ["memoriser", "oublier", "rechercher_souvenir", "relier"])

    def test_pas_de_changement_d_identite_ni_d_outil_d_interface(self):
        noms = self.noms()
        for interdit in ("identifier_interlocuteur", "ui_etat"):
            self.assertNotIn(interdit, noms)
        out = self.tool("ui_etat", {"activer": self.olivier})
        self.assertNotEqual(self.s.active, "")
        self.assertEqual(self.s.active, self.olivier)
        self.assertTrue("error" in out.lower() or "unknown" in out.lower() or "inconnu" in out.lower(), out)

    def test_aucun_schema_ne_contient_d_argument_d_identite(self):
        r = self.client.post("/mcp", json=rpc("tools/list"), headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"})
        for t in r.json()["result"]["tools"]:
            props = set(t["inputSchema"].get("properties", {}))
            self.assertFalse(props & {"owner", "principal", "ctx", "o", "who"}, (t["name"], props))


@unittest.skipUnless(HAVE, "mcp / starlette absents (lancer dans le conteneur)")
class GardeTests(unittest.TestCase):
    """Le garde d'authentification est la seule porte d'entrée : tout passe par lui AVANT les outils."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="httpgarde_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.svc = Service(self.s, Interlocuteurs(self.s))
        self.app = http_facade.make_app(self.s, self.svc, SECRET)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:7872")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def post(self, auth):
        h = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", "X-Memory-Principal": "k_x", "X-Memory-Name": "X"}
        if auth is not None:
            h["Authorization"] = auth
        return self.client.post("/mcp", json=rpc("tools/list"), headers=h)

    def test_sans_secret_ou_avec_un_mauvais_secret_401(self):
        for auth in (None, "", "Bearer", "Bearer faux", f"Bearer {SECRET}x", f"bearer {SECRET}", SECRET, f"Basic {SECRET}"):
            r = self.post(auth)
            self.assertEqual(r.status_code, 401, auth)
            self.assertNotIn("result", r.json())

    def test_bon_secret(self):
        self.assertEqual(self.post(f"Bearer {SECRET}").status_code, 200)

    def test_un_refus_ne_cree_aucune_personne_ni_souvenir(self):
        avant = (len(self.s.personnes.items), len(self.s.records))
        self.post("Bearer faux")
        self.assertEqual((len(self.s.personnes.items), len(self.s.records)), avant)

    def test_toutes_les_routes_sont_gardees(self):
        for methode, chemin in (("get", "/mcp"), ("post", "/mcp"), ("delete", "/mcp"), ("get", "/"), ("post", "/messages"), ("get", "/sse")):
            r = getattr(self.client, methode)(chemin)
            self.assertEqual(r.status_code, 401, (methode, chemin))


class DemarrageTests(unittest.TestCase):
    def test_secret_requis_pour_ouvrir_le_canal(self):
        src = (Path(__file__).resolve().parent.parent / "server.py").read_text("utf-8")
        self.assertIn("MEMORY_HTTP_SECRET", src)
        self.assertIn("trop court", src)                                  # un canal sans secret n'est jamais ouvert
        self.assertIn('host="127.0.0.1"', (Path(__file__).resolve().parent.parent / "http_facade.py").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
