"""Page d'administration des accès tiers : garde (hôte, jeton, origine), gestion des clés, profils, onglet. Bibliothèque standard seulement."""
import http.client
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import access_tab  # noqa: E402
from components.access import admin_ui, keys, profile  # noqa: E402

ORIGIN = "https://app.example.com"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSessions:
    def __init__(self):
        self.closed = []

    def request_close_key(self, kid):
        self.closed.append(kid)

    def stats(self):
        return [{"id": "k_1:s", "key": "k_1", "profile": "portail", "idle_s": 3}]


class AdminCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="accadm_"))
        self.config = {"mcp": {"params": {"servers": {"web": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer <Jeton>"}}}}},
                       "access": {"profiles": {"portail": {"description": "Portail", "servers": ["web"]}}}}
        self.store = keys.KeyStore(self.tmp / "data")
        self.profiles = profile.ConfigProfiles(lambda: self.config)
        self.sessions = FakeSessions()
        self.audit = []
        self.port = free_port()
        self.httpd, self.token = admin_ui.start(self.store, self.profiles, self.sessions, self.port,
                                                audit=lambda ev, **kw: self.audit.append((ev, kw)))

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def call(self, method, path, body=None, token="auto", origin="auto", host=None, ctype="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        host = host or f"127.0.0.1:{self.port}"
        hdr = {"Host": host}
        if token == "auto":
            token = self.token
        if token:
            hdr["X-Token"] = token
        if method == "POST":
            if origin == "auto":
                origin = f"http://{host}"
            if origin:
                hdr["Origin"] = origin
            if ctype:
                hdr["Content-Type"] = ctype
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=hdr)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw

    def create(self, **kw):
        corps = {"label": "Portail", "origins": [ORIGIN], "profile": "portail"}
        corps.update(kw)
        return self.call("POST", "/api/keys/create", corps)[1]


class GardeTests(AdminCase):
    def test_hote_inconnu_refuse(self):
        self.assertEqual(self.call("GET", "/api/keys", host="evil.example.com")[0], 403)
        self.assertEqual(self.call("GET", "/", host="evil.example.com")[0], 403)

    def test_jeton_obligatoire_sur_l_api_pas_sur_la_page(self):
        self.assertEqual(self.call("GET", "/api/keys", token=None)[0], 401)
        self.assertEqual(self.call("GET", "/api/keys", token="faux")[0], 401)
        self.assertEqual(self.call("GET", "/api/keys")[0], 200)
        status, page = self.call("GET", "/", token=None)
        self.assertEqual(status, 200)
        self.assertIn(self.token.encode(), page)                                        # la page reçoit le jeton pour ses appels

    def test_origine_obligatoire_et_identique_sur_les_ecritures(self):
        corps = {"label": "x", "origins": [ORIGIN]}
        self.assertEqual(self.call("POST", "/api/keys/create", corps, origin=None)[0], 403)         # absente : refusée
        self.assertEqual(self.call("POST", "/api/keys/create", corps, origin="http://evil.example.com")[0], 403)
        self.assertEqual(self.store.list(), [])
        self.assertEqual(self.call("POST", "/api/keys/create", corps)[0], 200)

    def test_json_obligatoire_et_corps_borne(self):
        self.assertEqual(self.call("POST", "/api/keys/create", {"label": "x"}, ctype="text/plain")[0], 415)
        self.assertEqual(self.call("POST", "/api/keys/create", {"label": "x" * 20000})[0], 400)

    def test_route_inconnue(self):
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)
        self.assertEqual(self.call("POST", "/api/nope", {"a": 1})[0], 404)

    def test_copie_de_la_cle_avec_repli_hors_page_securisee(self):
        # la page est servie en http sur 127.0.0.1 dans un cadre : navigator.clipboard y est absent, il faut un repli
        self.assertIn("isSecureContext", admin_ui.PAGE)
        self.assertIn("execCommand('copy')", admin_ui.PAGE)
        self.assertIn("Ctrl+C", admin_ui.PAGE)

    def test_la_page_ne_contient_pas_innerHTML(self):
        self.assertNotIn("innerHTML", admin_ui.PAGE)
        self.assertEqual(admin_ui.PAGE.count("</script>"), 1)


class ClesTests(AdminCase):
    def test_creation_renvoie_la_cle_une_seule_fois(self):
        r = self.create()
        self.assertTrue(r["ok"] and r["secret"].startswith("ak_"))
        liste = self.call("GET", "/api/keys")[1]
        self.assertEqual(len(liste), 1)
        self.assertNotIn(r["secret"], json.dumps(liste))
        self.assertNotIn("hash", json.dumps(liste))
        self.assertEqual(self.store.authenticate(r["secret"], ORIGIN)[0], "ok")                    # la clé fonctionne vraiment
        self.assertNotIn(r["secret"], json.dumps(self.audit))                                     # jamais dans le journal

    def test_creation_refusee(self):
        self.assertFalse(self.create(label="  ")["ok"])
        self.assertEqual(self.create(profile="inexistant")["erreur"], "profil inconnu")
        self.assertEqual(self.store.list(), [])

    def test_sans_profil_la_cle_n_a_aucun_outil(self):
        r = self.create(profile="")
        self.assertEqual(r["key"]["profile"], "")

    def test_modifier(self):
        kid = self.create()["key"]["id"]
        r = self.call("POST", "/api/keys/update", {"id": kid, "label": "Autre", "origins": ["https://b.example"], "voice": False,
                                                    "limits": {"rate_per_min": 5, "daily_turns": 9}})[1]
        self.assertTrue(r["ok"])
        self.assertEqual((r["key"]["label"], r["key"]["origins"], r["key"]["voice"]), ("Autre", ["https://b.example"], False))
        self.assertIn(kid, self.sessions.closed)                                                   # les sessions ouvertes sont fermées
        self.assertFalse(self.call("POST", "/api/keys/update", {"id": kid, "profile": "inexistant"})[1]["ok"])
        self.assertFalse(self.call("POST", "/api/keys/update", {"id": "k_zzz", "label": "x"})[1]["ok"])

    def test_revoquer_ferme_les_sessions_et_empeche_l_usage(self):
        r = self.create()
        kid, cle = r["key"]["id"], r["secret"]
        self.assertTrue(self.call("POST", "/api/keys/revoke", {"id": kid})[1]["ok"])
        self.assertEqual(self.store.authenticate(cle)[0], "revoked")
        self.assertIn(kid, self.sessions.closed)
        self.call("POST", "/api/keys/revoke", {"id": kid, "revoked": False})
        self.assertEqual(self.store.authenticate(cle)[0], "ok")

    def test_regenerer_la_cle_garde_l_identifiant_et_invalide_l_ancienne(self):
        r = self.create()
        kid, ancienne = r["key"]["id"], r["secret"]
        x = self.call("POST", "/api/keys/regenerate", {"id": kid})[1]
        self.assertTrue(x["ok"])
        self.assertEqual(x["key"]["id"], kid)                                             # même identité interne : la mémoire suit
        self.assertNotEqual(x["secret"], ancienne)
        self.assertEqual(self.store.authenticate(ancienne)[0], "unknown")
        self.assertEqual(self.store.authenticate(x["secret"], ORIGIN)[0], "ok")
        self.assertIn(kid, self.sessions.closed)                                          # les conversations ouvertes repartent
        self.assertNotIn(x["secret"], json.dumps(self.audit))                             # jamais dans le journal
        self.assertEqual([ev for ev, _ in self.audit][-1], "key_regenerated")

    def test_regenerer_un_acces_inconnu(self):
        x = self.call("POST", "/api/keys/regenerate", {"id": "k_zzz"})[1]
        self.assertFalse(x["ok"])
        self.assertNotIn("secret", x)

    def test_regenerer_exige_le_jeton_et_l_origine(self):
        r = self.create()
        self.assertEqual(self.call("POST", "/api/keys/regenerate", {"id": r["key"]["id"]}, token=None)[0], 401)
        self.assertEqual(self.call("POST", "/api/keys/regenerate", {"id": r["key"]["id"]}, origin=None)[0], 403)
        self.assertEqual(self.store.authenticate(r["secret"], ORIGIN)[0], "ok")           # rien n'a changé

    def test_la_liste_ne_renvoie_jamais_la_cle_regeneree(self):
        r = self.create()
        x = self.call("POST", "/api/keys/regenerate", {"id": r["key"]["id"]})[1]
        self.assertNotIn(x["secret"], json.dumps(self.call("GET", "/api/keys")[1]))

    def test_la_page_propose_la_regeneration(self):
        self.assertIn("/api/keys/regenerate", admin_ui.PAGE)
        self.assertIn("🔄", admin_ui.PAGE)

    def test_supprimer(self):
        r = self.create()
        self.assertTrue(self.call("POST", "/api/keys/delete", {"id": r["key"]["id"]})[1]["ok"])
        self.assertEqual(self.store.authenticate(r["secret"])[0], "unknown")
        self.assertFalse(self.call("POST", "/api/keys/delete", {"id": r["key"]["id"]})[1]["ok"])

    def test_audit_des_actions(self):
        r = self.create()
        self.call("POST", "/api/keys/revoke", {"id": r["key"]["id"]})
        self.assertEqual([ev for ev, _ in self.audit], ["key_created", "key_revoked"])


class ProfilsTests(AdminCase):
    def test_liste_sans_secrets(self):
        p = self.call("GET", "/api/profils")[1]
        self.assertEqual([x["name"] for x in p], ["portail"])
        self.assertEqual(p[0]["servers"], [{"name": "web", "type": "http"}])
        self.assertNotIn("Jeton", json.dumps(p))
        self.assertNotIn("Bearer", json.dumps(p))

    def test_profil_invalide_signale_sans_planter(self):
        self.config["access"]["profiles"]["casse"] = {"servers": ["fantome"]}
        p = {x["name"]: x for x in self.call("GET", "/api/profils")[1]}
        self.assertIn("erreur", p["casse"])
        self.assertNotIn("erreur", p["portail"])

    def test_les_profils_modifies_dans_les_parametres_apparaissent_sans_redemarrage(self):
        self.config["access"]["profiles"]["nouveau"] = {"description": "Ajouté", "servers": []}
        self.assertEqual([x["name"] for x in self.call("GET", "/api/profils")[1]], ["nouveau", "portail"])
        kid = self.create(profile="nouveau")["key"]["id"]                       # et une clé peut le choisir tout de suite
        self.assertEqual(self.store.get(kid)["profile"], "nouveau")
        del self.config["access"]["profiles"]["nouveau"]
        self.assertEqual(self.create(profile="nouveau")["erreur"], "profil inconnu")

    def test_la_page_ne_parle_plus_de_fichiers_de_profil(self):
        self.assertNotIn("configs/mcp", admin_ui.PAGE)
        self.assertIn("Paramètres", admin_ui.PAGE)

    def test_sessions(self):
        self.assertEqual(self.call("GET", "/api/sessions")[1][0]["key"], "k_1")


class OngletTests(unittest.TestCase):
    def test_port_seulement_si_active(self):
        self.assertIsNone(access_tab.admin_port({}))
        self.assertIsNone(access_tab.admin_port({"access": {"enabled": False}}))
        self.assertEqual(access_tab.admin_port({"access": {"enabled": True}}), 7871)
        self.assertEqual(access_tab.admin_port({"access": {"enabled": True, "admin_port": 9000}}), 9000)
        self.assertIsNone(access_tab.admin_port({"access": {"enabled": True, "admin_port": "abc"}}))
        self.assertIsNone(access_tab.admin_port({"access": {"enabled": True, "admin_port": 70000}}))

    def test_cadre_local_uniquement(self):
        html = access_tab.html(7871)
        self.assertIn("http://127.0.0.1:7871/", html)
        self.assertIn("iframe", html)
        self.assertIn("aria_access_frame", access_tab.js_reload())


if __name__ == "__main__":
    unittest.main()
