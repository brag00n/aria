"""Page d'administration des agents : garde (hôte, jeton, origine), gestion des clés, profils, onglet. Bibliothèque standard seulement."""
import http.client
import json
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import agents_tab, audit_retention, echanges  # noqa: E402
from components.agents import admin_ui, keys, profile  # noqa: E402

ORIGIN = "https://app.example.com"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSessions:
    def __init__(self):
        self.closed = []
        self.closed_sessions = []
        self.open = {"k_1:s"}                      # sessions « ouvertes » du faux registre

    def request_close_key(self, kid):
        self.closed.append(kid)

    def request_close_session(self, sid):
        if sid not in self.open:
            return False
        self.closed_sessions.append(sid)
        return True

    def stats(self):
        return [{"id": "k_1:s", "key": "k_1", "profile": "portail", "idle_s": 3}]


class AdminCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="accadm_"))
        self.config = {"mcp": {"params": {"servers": {"web": {"url": "http://x/mcp", "headers": {"Authorization": "Bearer <Jeton>"}}}}},
                       "access": {"profiles": {"portail": {"description": "Portail", "servers": ["web"]}}}}
        self.store = keys.KeyStore(self.tmp / "data")
        self.persistes = []                                   # chaque appel de persist() : une copie des profils écrits
        self.persist_echoue = False
        self.profiles = profile.ConfigProfiles(lambda: self.config, self._persist)
        self.sessions = FakeSessions()
        self.audit = []
        self.effaces = []                                   # appels de l'effacement des traces (simulé : la mémoire n'est pas démarrée ici)
        echanges.configure(self.tmp / "audit", enabled=True, log_text=True, retention_days=90)
        self.port = free_port()
        self.httpd, self.token = admin_ui.start(self.store, self.profiles, self.sessions, self.port,
                                                audit=lambda ev, **kw: self.audit.append((ev, kw)),
                                                effacer_traces=self._effacer)

    def _persist(self, profils):
        if self.persist_echoue:
            raise OSError("disque plein")
        self.persistes.append(json.loads(json.dumps(profils)))

    def _effacer(self, qui, ids):
        if getattr(self, "panne_memoire", False):
            raise RuntimeError("mémoire indisponible")
        self.effaces.append((qui, list(ids)))
        return {"souvenirs": 2, "personnes": 1}

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
        self.assertEqual([k["id"] for k in self.store.list()], ["aria"])
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
        self.assertEqual([a["id"] for a in liste if a["id"] != "aria"], [r["key"]["id"]])
        self.assertNotIn(r["secret"], json.dumps(liste))
        self.assertNotIn("hash", json.dumps(liste))
        self.assertEqual(self.store.authenticate(r["secret"], ORIGIN)[0], "ok")                    # la clé fonctionne vraiment
        self.assertNotIn(r["secret"], json.dumps(self.audit))                                     # jamais dans le journal

    def test_creation_refusee(self):
        self.assertFalse(self.create(label="  ")["ok"])
        self.assertEqual(self.create(profile="inexistant")["erreur"], "profil inconnu")
        self.assertEqual([k["id"] for k in self.store.list()], ["aria"])                          # rien de créé, hors l'agent par défaut

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
        self.assertNotIn("Paramètres d", admin_ui.PAGE)                                    # les profils ne se gèrent plus dans les Paramètres d'Aria

    def test_l_onglet_s_appelle_profils(self):
        self.assertIn('data-t="prof">Profils</button>', admin_ui.PAGE)
        self.assertNotIn("Profils MCP", admin_ui.PAGE)
        for mot in ("Nouveau profil", "/api/profils/save", "/api/profils/delete", "tabsOf", "Conversation", "Outils MCP", "Message système"):
            self.assertIn(mot, admin_ui.PAGE)

    def test_sessions(self):
        self.assertEqual(self.call("GET", "/api/sessions")[1][0]["key"], "k_1")


class EditionProfilsTests(AdminCase):
    """Création, modification et suppression des profils depuis la page : validés, écrits dans la config, sessions rafraîchies."""

    def sauver(self, name, profil, create=True):
        return self.call("POST", "/api/profils/save", {"name": name, "create": create, "profile": profil})[1]

    def test_creation_ecrit_dans_la_config_et_dans_le_fichier(self):
        r = self.sauver("neuf", {"description": "Neuf", "system_message": "Tu es neuf.", "voice": False, "servers": ["web"],
                                 "conversation": {"enabled": True, "open_phrases": ["salut robot"]}})
        self.assertTrue(r["ok"], r)
        self.assertEqual(self.config["agents"]["profiles"]["neuf"]["conversation"]["open_phrases"], ["salut robot"])
        self.assertEqual(self.persistes[-1]["neuf"]["voice"], False)
        self.assertIn("portail", self.persistes[-1])                                       # les autres profils sont conservés
        self.assertIn("neuf", [x["name"] for x in self.call("GET", "/api/profils")[1]])

    def test_creation_d_un_profil_existant_refusee(self):
        r = self.sauver("portail", {"servers": []})
        self.assertFalse(r["ok"])
        self.assertIn("existe déjà", r["erreur"])
        self.assertEqual(self.persistes, [])

    def test_modification_d_un_profil_inconnu_refusee(self):
        r = self.sauver("fantome", {"servers": []}, create=False)
        self.assertIn("introuvable", r["erreur"])

    def test_modification_conserve_les_autres_cles_de_la_section(self):
        self.config["access"]["enabled"] = True
        self.sauver("portail", {"description": "Changé", "servers": ["web"]}, create=False)
        self.assertEqual((self.config["agents"]["enabled"], self.config["agents"]["profiles"]["portail"]["description"]), (True, "Changé"))
        self.assertNotIn("access", self.config)                                           # écrit sous le nouveau nom

    def test_profil_invalide_refuse_sans_rien_modifier(self):
        avant = json.loads(json.dumps(self.config))
        for name, data, mot in [("a", {"servers": ["fantome"]}, "fantome"), ("b", {"voix": True}, "voix"), ("c", {"llm": "Inconnu"}, "Inconnu"),
                                ("d", {"conversation": {"enabled": "oui"}}, "enabled"), ("e", {"voice": "oui"}, "voice"), ("x" * 61, {}, "nom")]:
            r = self.sauver(name, data)
            self.assertFalse(r["ok"], (name, r))
            self.assertIn(mot, r["erreur"])
        self.assertEqual(self.config, avant)
        self.assertEqual(self.persistes, [])

    def test_nom_invalide(self):
        for nom in ("", "a b", "../x", "é"):
            self.assertFalse(self.sauver(nom, {})["ok"], nom)

    def test_un_llm_local_est_accepte_a_l_enregistrement(self):
        self.config["llm"] = {"Llm_Local": {"params": {"backend_type": "local"}}}
        self.assertTrue(self.sauver("loc", {"llm": "Llm_Local"})["ok"])

    def test_echec_d_ecriture_laisse_la_memoire_intacte(self):
        avant = json.loads(json.dumps(self.config))
        self.persist_echoue = True
        r = self.sauver("neuf", {"servers": []})
        self.assertFalse(r["ok"])
        self.assertIn("écriture", r["erreur"])
        self.assertEqual(self.config, avant)                                              # jamais un profil actif en mémoire mais perdu au redémarrage
        self.assertNotIn("neuf", self.profiles.names())

    def test_echec_d_ecriture_a_la_suppression(self):
        self.persist_echoue = True
        self.assertFalse(self.call("POST", "/api/profils/delete", {"name": "portail"})[1]["ok"])
        self.assertIn("portail", self.profiles.names())

    def test_modification_ferme_les_sessions_des_agents_du_profil_seulement(self):
        k1 = self.create(profile="portail")["key"]["id"]
        self.sauver("autre", {"servers": []})
        k2 = self.create(label="B", profile="autre")["key"]["id"]
        r = self.sauver("portail", {"description": "v2", "servers": ["web"]}, create=False)
        self.assertEqual(r["agents"], 1)
        self.assertEqual(self.sessions.closed, [k1])
        self.assertNotIn(k2, self.sessions.closed)

    def test_suppression(self):
        self.sauver("jetable", {"servers": []})
        r = self.call("POST", "/api/profils/delete", {"name": "jetable"})[1]
        self.assertTrue(r["ok"])
        self.assertNotIn("jetable", self.config["agents"]["profiles"])
        self.assertNotIn("jetable", self.persistes[-1])
        self.assertIn("portail", self.config["agents"]["profiles"])                         # les autres profils restent
        self.assertIn("portail", self.persistes[-1])

    def test_suppression_refusee_si_un_agent_l_utilise(self):
        self.create(profile="portail")
        r = self.call("POST", "/api/profils/delete", {"name": "portail"})[1]
        self.assertFalse(r["ok"])
        self.assertIn("Portail", r["erreur"])
        self.assertIn("portail", self.profiles.names())

    def test_suppression_d_un_profil_inconnu(self):
        self.assertIn("introuvable", self.call("POST", "/api/profils/delete", {"name": "fantome"})[1]["erreur"])

    def test_corps_non_objet_refuse(self):
        for data in (None, [], "texte", 3):
            self.assertFalse(self.sauver("z", data)["ok"], data)

    def test_textes_trop_longs_refuses(self):
        self.assertFalse(self.sauver("z", {"description": "x" * 201})["ok"])
        self.assertFalse(self.sauver("z", {"system_message": "x" * 12001})["ok"])
        self.assertTrue(self.sauver("z", {"system_message": "x" * 12000})["ok"])

    def test_les_routes_d_ecriture_exigent_jeton_et_origine(self):
        corps = {"name": "z", "create": True, "profile": {}}
        self.assertEqual(self.call("POST", "/api/profils/save", corps, token=None)[0], 401)
        self.assertEqual(self.call("POST", "/api/profils/save", corps, origin="https://evil.example.com")[0], 403)
        self.assertEqual(self.call("POST", "/api/profils/delete", {"name": "portail"}, token=None)[0], 401)
        self.assertNotIn("z", self.profiles.names())

    def test_options_pour_l_editeur_sans_secrets(self):
        self.config["llm"] = {"Llm_A": {"params": {"backend_type": "anthropic", "model_name": "m1", "apikey": "SECRET-CLE", "base_url": "http://x"}},
                              "Llm_L": {"params": {"backend_type": "local", "model_name": "m2"}}}
        o = self.call("GET", "/api/profils/options")[1]
        self.assertEqual({m["name"]: m["local"] for m in o["llm"]}, {"Llm_A": False, "Llm_L": True})
        self.config["access"]["private_servers"] = ["prive"]
        self.config["mcp"]["params"]["servers"]["prive"] = {"url": "http://prive/mcp"}
        o = self.call("GET", "/api/profils/options")[1]
        self.assertEqual(o["servers"], [{"name": "web", "type": "http"}])                   # un serveur privé n'est jamais proposé
        self.assertIn("open_phrases", o["conversation"])
        self.assertNotIn("SECRET-CLE", json.dumps(o))
        self.assertNotIn("Bearer", json.dumps(o))

    def test_la_liste_expose_le_profil_a_modifier(self):
        p = {x["name"]: x for x in self.call("GET", "/api/profils")[1]}["portail"]
        self.assertEqual(p["raw"]["servers"], ["web"])
        self.assertEqual(p["conversation"], None)


class JournalTests(AdminCase):
    """Page « Journal » : volumes, purge manuelle et ciblée, journal des purges. Jamais le contenu des conversations."""

    def ecrire(self, src="aria", who="Olivier", conv="c1", q="Question très secrète", jours=1):
        import time
        e = echanges.Echange(src, who, conv, query=q)
        e.ts = time.time() - jours * 86400
        echanges.write(e, "Réponse très secrète", "ok")
        return e

    def test_lecture_sans_aucun_contenu(self):
        self.ecrire()
        status, j = self.call("GET", "/api/journal")
        brut = json.dumps(j, ensure_ascii=False)
        self.assertEqual(status, 200)
        self.assertNotIn("secrète", brut)
        self.assertEqual(sum(v["lignes"] for v in j["volumes"]), 1)
        self.assertEqual((j["defaut_jours"], j["actif"], j["texte"]), (90, True, True))

    def test_la_lecture_exige_le_jeton_et_l_ecriture_l_origine(self):
        self.assertEqual(self.call("GET", "/api/journal", token=None)[0], 401)
        self.assertEqual(self.call("GET", "/api/journal", host="evil.example.com")[0], 403)
        self.assertEqual(self.call("POST", "/api/journal/purge", {"conversation": "c1"}, origin=None)[0], 403)
        self.assertEqual(self.call("POST", "/api/journal/purge", {"conversation": "c1"}, token=None)[0], 401)
        self.assertEqual(self.call("POST", "/api/journal/purge-age", {}, origin="http://evil.example.com")[0], 403)

    def test_purge_ciblee_par_conversation_et_journal_des_purges(self):
        a, b = self.ecrire(conv="c1"), self.ecrire(conv="c2")
        r = self.call("POST", "/api/journal/purge", {"conversation": "c1"})[1]
        self.assertEqual((r["ok"], r["lignes"]), (True, 1))
        restants = [l["id"] for _, p in echanges.files() for l in echanges.read_lines(p)]
        self.assertEqual(restants, [b.id])
        purges = self.call("GET", "/api/journal")[1]["purges"]
        self.assertEqual((purges[0]["par"], purges[0]["lignes"]), ("admin", 1))
        self.assertNotIn("secrète", json.dumps(purges))
        self.assertIn(("audit_purge", {"lignes": 1, "fichiers": 0}), self.audit)

    def test_aucun_critere_refuse_et_rien_n_est_supprime(self):
        self.ecrire()
        for corps in ({}, {"conversation": ""}, {"date_avant": "  "}):
            r = self.call("POST", "/api/journal/purge", corps)[1]
            self.assertFalse(r["ok"], corps)
        self.assertEqual(len([1 for _, p in echanges.files() for _ in echanges.read_lines(p)]), 1)

    def test_date_invalide_refusee(self):
        self.assertFalse(self.call("POST", "/api/journal/purge", {"date_avant": "demain"})[1]["ok"])

    def test_purge_d_une_personne_efface_aussi_les_traces_de_la_memoire(self):
        e = self.ecrire(who="Léa")
        self.ecrire(who="Paul")
        r = self.call("POST", "/api/journal/purge", {"personne": "Léa"})[1]
        self.assertEqual((r["lignes"], r["traces"]), (1, {"souvenirs": 2, "personnes": 1}))
        self.assertEqual(self.effaces, [("Léa", [e.id])])
        purges = self.call("GET", "/api/journal")[1]["purges"]
        self.assertEqual(sum(p["traces"] for p in purges), 3)
        self.assertNotIn("Léa", json.dumps(purges, ensure_ascii=False))

    def test_les_autres_purges_ne_touchent_pas_aux_traces(self):
        self.ecrire(conv="c1")
        self.call("POST", "/api/journal/purge", {"conversation": "c1"})
        self.call("POST", "/api/journal/purge", {"acces": "aria"})
        self.assertEqual(self.effaces, [])

    def test_une_memoire_indisponible_n_annule_pas_la_purge_du_journal(self):
        self.ecrire(who="Léa")
        self.panne_memoire = True
        r = self.call("POST", "/api/journal/purge", {"personne": "Léa"})[1]
        self.assertTrue(r["ok"])
        self.assertEqual(r["lignes"], 1)
        self.assertIn("erreur", r["traces"])                                # signalé, pas silencieux
        self.assertEqual([l for _, p in echanges.files() for l in echanges.read_lines(p)], [])

    def test_purge_selon_la_duree_de_chaque_acces(self):
        vue = self.create(label="Robot")["key"]
        self.store.update(vue["id"], retention_days=7)
        court = self.ecrire(src=f"{vue['id']}:Robot", who="Robot", jours=20)
        garde = self.ecrire(jours=20)
        r = self.call("POST", "/api/journal/purge-age", {})[1]
        self.assertEqual(r["lignes"], 1)
        restants = [l["id"] for _, p in echanges.files() for l in echanges.read_lines(p)]
        self.assertEqual(restants, [garde.id])
        self.assertNotIn(court.id, restants)

    def test_duree_par_acces_modifiable_depuis_la_page_des_cles(self):
        vue = self.create(label="Robot", retention_days=14)["key"]
        self.assertEqual(vue["retention_days"], 14)
        self.call("POST", "/api/keys/update", {"id": vue["id"], "retention_days": 30})
        self.assertEqual(self.store.retention_of(vue["id"]), 30)
        j = self.call("GET", "/api/journal")[1]
        self.assertEqual([a["retention_days"] for a in j["acces"] if a["id"] != "aria"], [30])

    def test_option_publier_ses_liens_creation_modification_et_page(self):
        vue = self.create(label="Robot", publish_links=True)["key"]
        self.assertTrue(vue["publish_links"])
        self.call("POST", "/api/keys/update", {"id": vue["id"], "publish_links": False})
        self.assertFalse(self.store.get(vue["id"])["publish_links"])
        self.call("POST", "/api/keys/update", {"id": vue["id"], "publish_links": True})
        self.assertTrue(self.store.get(vue["id"])["publish_links"])
        self.assertIn("publish_links", admin_ui.PAGE)
        self.assertIn("jamais de lien entre d", admin_ui.PAGE)

    def test_changer_l_option_ferme_les_sessions_de_la_cle(self):
        vue = self.create(label="Robot")["key"]
        self.call("POST", "/api/keys/update", {"id": vue["id"], "publish_links": True})
        self.assertIn(vue["id"], self.sessions.closed)

    def test_option_publier_les_infos_de_son_interlocuteur(self):
        vue = self.create(label="Robot", publish_speaker=True)["key"]
        self.assertTrue(vue["publish_speaker"])
        self.call("POST", "/api/keys/update", {"id": vue["id"], "publish_speaker": False})
        self.assertFalse(self.store.get(vue["id"])["publish_speaker"])
        self.sessions.closed.clear()
        self.call("POST", "/api/keys/update", {"id": vue["id"], "publish_speaker": True})
        self.assertTrue(self.store.get(vue["id"])["publish_speaker"])
        self.assertIn(vue["id"], self.sessions.closed)                       # les sessions ouvertes repartent avec le nouveau droit
        self.assertIn("publish_speaker", admin_ui.PAGE)
        self.assertIn("PAS authentifiée", admin_ui.PAGE)                     # l'avertissement est dans la page

    def test_fermer_une_session_depuis_la_page(self):
        r = self.call("POST", "/api/sessions/close", {"id": "k_1:s"})[1]
        self.assertTrue(r["ok"])
        self.assertEqual(self.sessions.closed_sessions, ["k_1:s"])
        self.assertIn(("session_closed", {"id": "k_1:s", "ok": True}), self.audit)

    def test_fermer_une_session_inconnue_ne_ferme_rien(self):
        for sid in ("k_1:autre", "", "k_2:s", "../k_1:s", "k_1:s ", "x" * 500):
            r = self.call("POST", "/api/sessions/close", {"id": sid})[1]
            self.assertFalse(r["ok"], sid)
        self.assertEqual(self.sessions.closed_sessions, [])

    def test_fermer_une_session_exige_jeton_origine_et_hote(self):
        corps = {"id": "k_1:s"}
        self.assertEqual(self.call("POST", "/api/sessions/close", corps, token=None)[0], 401)
        self.assertEqual(self.call("POST", "/api/sessions/close", corps, origin=None)[0], 403)
        self.assertEqual(self.call("POST", "/api/sessions/close", corps, origin="http://evil.example.com")[0], 403)
        self.assertEqual(self.call("POST", "/api/sessions/close", corps, host="evil.example.com")[0], 403)
        self.assertEqual(self.sessions.closed_sessions, [])

    def test_la_page_a_le_bouton_de_fermeture_sans_innerhtml(self):
        self.assertIn("/api/sessions/close", admin_ui.PAGE)
        self.assertIn("confirm(", admin_ui.PAGE)
        self.assertNotIn("innerHTML", admin_ui.PAGE)

    def test_droits_a_la_creation_et_a_la_modification(self):
        vue = self.create(label="Robot", rights={"local_llm": True, "no_limits": True, "inconnu": True})["key"]
        self.assertEqual([n for n, v in vue["rights"].items() if v], ["local_llm", "no_limits"])
        self.call("POST", "/api/keys/update", {"id": vue["id"], "rights": {"all_mcp": True}})
        self.assertEqual([n for n, v in self.store.get(vue["id"])["rights"].items() if v], ["all_mcp"])
        self.assertIn(vue["id"], self.sessions.closed)                                         # les sessions ouvertes repartent avec les nouveaux droits

    def test_droits_absurdes_ne_donnent_rien(self):
        vue = self.create(label="Robot", rights="tout")["key"]
        self.assertFalse(any(vue["rights"].values()))
        for absurde in ({"local_llm": "true"}, {"local_llm": 1}, [1], 5, None):
            self.call("POST", "/api/keys/update", {"id": vue["id"], "rights": absurde})
            self.assertFalse(any(self.store.get(vue["id"])["rights"].values()), repr(absurde))

    def test_l_agent_par_defaut_est_visible_et_verrouille_dans_la_liste(self):
        liste = self.call("GET", "/api/keys")[1]
        aria = next(a for a in liste if a["id"] == "aria")
        self.assertTrue(aria["system"] and all(aria["rights"].values()))
        self.assertNotIn("hash", json.dumps(aria))

    def test_l_agent_par_defaut_ne_se_supprime_ni_ne_se_revoque_par_la_page(self):
        r = self.call("POST", "/api/keys/delete", {"id": "aria"})[1]
        self.assertFalse(r["ok"])
        self.assertIn("par défaut", r["erreur"])
        r = self.call("POST", "/api/keys/revoke", {"id": "aria", "revoked": True})[1]
        self.assertFalse(r["ok"])
        self.assertIsNotNone(self.store.get("aria"))
        self.assertFalse(self.store.get("aria")["revoked"])

    def test_les_droits_de_l_agent_par_defaut_ne_se_modifient_pas_par_la_page(self):
        self.call("POST", "/api/keys/update", {"id": "aria", "rights": {}, "publish_links": False, "publish_speaker": False, "profile": "portail"})
        aria = self.store.get("aria")
        self.assertTrue(all(aria["rights"].values()))
        self.assertEqual(aria["profile"], "aria")

    def test_on_peut_regenerer_la_cle_de_l_agent_par_defaut(self):
        r = self.call("POST", "/api/keys/regenerate", {"id": "aria"})[1]
        self.assertTrue(r["ok"] and r["secret"].startswith("ak_"))
        self.assertEqual(self.store.authenticate(r["secret"])[0], "ok")

    def test_la_page_grise_les_droits_de_l_agent_par_defaut(self):
        # chaque case de droit (les quatre nouveaux et les deux historiques) est cochée ET désactivée pour l'agent système, pas pour les autres
        for ligne in ("const publish=", "const speaker=", "R[n]=el('input'"):
            ext = next(l for l in admin_ui.PAGE.splitlines() if ligne in l)
            self.assertIn("checked:sys||", ext, ligne)
            self.assertIn("disabled:sys", ext, ligne)
            self.assertNotIn("disabled:false", ext, ligne)
        self.assertIn("agent par défaut", admin_ui.PAGE)
        for n in ("local_llm", "all_mcp", "global_speaker", "no_limits"):
            self.assertIn(n, admin_ui.PAGE)
        self.assertNotIn("innerHTML", admin_ui.PAGE)

    def test_la_page_a_son_onglet_sans_innerhtml(self):
        self.assertIn('data-t="jrn"', admin_ui.PAGE)
        self.assertIn("/api/journal/purge", admin_ui.PAGE)
        self.assertNotIn("innerHTML", admin_ui.PAGE)
        self.assertEqual(admin_ui.PAGE.count("</script>"), 1)
        self.assertIn("Cette page n", admin_ui.PAGE)                       # rappel : jamais le contenu des conversations


class OngletTests(unittest.TestCase):
    def test_port_seulement_si_active(self):
        self.assertIsNone(agents_tab.admin_port({}))
        self.assertIsNone(agents_tab.admin_port({"access": {"enabled": False}}))
        self.assertEqual(agents_tab.admin_port({"access": {"enabled": True}}), 7871)
        self.assertEqual(agents_tab.admin_port({"access": {"enabled": True, "admin_port": 9000}}), 9000)
        self.assertIsNone(agents_tab.admin_port({"access": {"enabled": True, "admin_port": "abc"}}))
        self.assertIsNone(agents_tab.admin_port({"access": {"enabled": True, "admin_port": 70000}}))

    def test_cadre_local_uniquement(self):
        html = agents_tab.html(7871)
        self.assertIn("http://127.0.0.1:7871/", html)
        self.assertIn("iframe", html)
        self.assertIn("aria_agents_frame", agents_tab.js_reload())


if __name__ == "__main__":
    unittest.main()
