"""Tests de la page de gestion (API sur un port libre) et du script d'analyse."""
import http.client
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import analyze  # noqa: E402
import store as store_mod  # noqa: E402
import ui  # noqa: E402
from activity import Activity  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UiCase(unittest.TestCase):
    token = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memui_"))
        self.activity = Activity(self.tmp / "logs")
        self.store = store_mod.Store(self.tmp, "default", activity=self.activity, lock_instance=False)
        self.store.active = "p0"
        self.port = free_port()
        self.httpd = ui.start(self.store, self.activity, self.port, self.token)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdr = {"Host": host or f"127.0.0.1:{self.port}"}
        hdr.update(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            hdr.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdr)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw


class UiTests(UiCase):
    def test_page_servie(self):
        status, raw = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"textContent" if False else b"<title>M", raw)

    def test_la_page_n_utilise_pas_innerhtml(self):
        self.assertNotIn(b"innerHTML", ui.PAGE.encode("utf-8"))

    def test_ajout_liste_modification_oubli_restauration(self):
        s, r = self.call("POST", "/api/add", {"texte": "Marion préfère 19 degrés", "type": "preference", "tags": "famille"})
        self.assertEqual((s, r["statut"]), (200, "added"))
        rid = r["id"]
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual(rows[0]["texte"], "Marion préfère 19 degrés")
        self.assertEqual(rows[0]["src"], "ui")
        self.assertEqual(rows[0]["etat"], "actif")
        self.assertTrue(self.call("POST", "/api/update", {"id": rid, "texte": "Marion préfère 20 degrés"})[1]["ok"])
        _, found = self.call("GET", "/api/memories?q=" + "20+degr%C3%A9s")
        self.assertEqual(found[0]["id"], rid)
        self.assertTrue(self.call("POST", "/api/pin", {"id": rid})[1]["ok"])
        self.assertEqual(self.call("GET", "/api/memories")[1][0]["etat"], "protege")
        self.assertTrue(self.call("POST", "/api/delete", {"id": rid})[1]["ok"])
        self.assertEqual(self.call("GET", "/api/memories")[1], [])
        self.assertEqual(self.call("GET", "/api/archive")[1][0]["why"], "ui")
        self.assertTrue(self.call("POST", "/api/restore", {"id": rid})[1]["ok"])
        self.assertEqual(len(self.call("GET", "/api/memories")[1]), 1)

    def test_stats_et_journal(self):
        self.call("POST", "/api/add", {"texte": "un souvenir de test"})
        self.assertEqual(self.call("GET", "/api/stats")[1]["total"], 1)
        events = [e["ev"] for e in self.call("GET", "/api/log")[1]]
        self.assertIn("ui", events)

    def test_hote_non_autorise_refuse(self):
        status, _ = self.call("GET", "/api/memories", host="evil.example.com")
        self.assertEqual(status, 403)

    def test_post_avec_mauvaise_origine_refuse(self):
        status, _ = self.call("POST", "/api/add", {"texte": "x y z"}, headers={"Origin": "http://evil.example.com"})
        self.assertEqual(status, 403)
        self.assertEqual(len(self.store.records), 0)

    def test_post_sans_json_refuse(self):
        status, _ = self.call("POST", "/api/add", b"texte=abc", headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_corps_trop_gros_refuse(self):
        status, _ = self.call("POST", "/api/add", {"texte": "a" * 50_000})
        self.assertEqual(status, 400)

    def test_texte_html_stocke_tel_quel_sans_etre_interprete(self):
        evil = "<script>alert(1)</script> chatte"
        self.call("POST", "/api/add", {"texte": evil})
        _, rows = self.call("GET", "/api/memories")
        self.assertEqual(rows[0]["texte"], evil)        # JSON pur ; la page l'affiche avec textContent
        self.assertNotIn(b"innerHTML", ui.PAGE.encode("utf-8"))

    def test_route_inconnue(self):
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)


class UiTokenTests(UiCase):
    token = "secret-test"

    def test_jeton_exige_pour_l_api(self):
        self.assertEqual(self.call("GET", "/api/memories")[0], 401)
        self.assertEqual(self.call("GET", "/api/memories", headers={"X-Token": "secret-test"})[0], 200)
        self.assertEqual(self.call("GET", "/")[0], 200)       # la page elle-même se charge, ses appels portent le jeton


class AnalyzeTests(unittest.TestCase):
    def test_resume(self):
        tmp = Path(tempfile.mkdtemp(prefix="memana_"))
        try:
            act = Activity(tmp / "logs")
            s = store_mod.Store(tmp, "default", activity=act, lock_instance=False)
            s.active = "p0"
            _, a = s.add("Marion préfère 19 degrés", "préférence")
            s.add("Le compteur Linky est l'équipement 270")
            s.add("Marion préfère 19 degrés")
            s.search("Marion")
            s.search("zzzzqqq")
            summary = analyze.summarize(tmp)
            self.assertEqual(summary["souvenirs"], 2)
            self.assertEqual(summary["ajouts"], 3)
            self.assertEqual(summary["doublons"], 1)
            self.assertEqual(summary["recherches_llm"], 2)
            self.assertEqual(summary["plus_rappeles"][0][0], a["id"])
            self.assertEqual(len(summary["jamais_rappeles"]), 1)
            self.assertEqual(len(summary["recherches_en_repli"]), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dossier_vide(self):
        tmp = Path(tempfile.mkdtemp(prefix="memana_"))
        try:
            summary = analyze.summarize(tmp)
            self.assertEqual(summary["souvenirs"], 0)
            self.assertEqual(summary["recherches_llm"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
