"""Accès tiers qui « publie les informations de son interlocuteur » (canal /mcp-i) : l'accès agit au nom de la personne qui lui parle, avec les droits
d'Aria, mais l'interlocuteur est propre à sa SESSION et n'est jamais `store.active` (celui d'Aria). Ses observations propres restent privées.
Aucune authentification de la personne n'est faite (à venir) : l'application ou le LLM la désigne."""
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store as store_mod  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402

try:
    import httpx  # noqa: F401
    from starlette.testclient import TestClient
    import http_facade
    HAVE = True
except ImportError:
    HAVE = False

SECRET = "secret-de-test-assez-long"


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        self.t += 1
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="interlocuteur_tiers_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.i = Interlocuteurs(self.s)
        self.svc = Service(self.s, self.i)
        self.i.identifier("Olivier", "Cousinier")                      # interlocuteur d'ARIA
        self.olivier = self.s.active
        self.i.relier("Tiphaine", "conjoint", "moi")
        self.s.add("Olivier aime la science-fiction", "preference")
        self.bamboo = self.s.personnes.ensure_system("k_bamboo", "Bamboo")["id"]
        self.autre = self.s.personnes.ensure_system("k_autre", "Autre")["id"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rec(self, mot):
        return next((r for r in self.s.records.values() if mot in r["txt"]), None)

    def personne(self, nom):
        return next((p for p in self.s.personnes.items if p["nom"] == nom), None)


class ServiceTests(Base):
    """Cœur (sans transport) : l'interlocuteur est un paramètre, jamais `store.active`."""

    def test_memoriser_va_a_l_interlocuteur_pas_au_systeme(self):
        tiph = self.personne("Tiphaine")["id"]
        out = self.svc.memoriser_pour(tiph, "Tiphaine aime le thé", "preference")
        self.assertTrue(out.startswith("OK"), out)
        self.assertEqual(self.rec("thé")["o"], tiph)
        self.assertEqual(self.s.active, self.olivier)                         # l'interlocuteur d'Aria n'a pas bougé

    def test_sans_interlocuteur_rien_n_est_memorise(self):
        self.assertIn("Interlocuteur inconnu", self.svc.memoriser_pour("", "Un fait", "fait"))
        self.assertIn("Interlocuteur inconnu", self.svc.memoriser_pour("p999", "Un fait", "fait"))
        self.assertIsNone(self.rec("Un fait"))

    def test_un_systeme_n_est_pas_un_interlocuteur(self):
        self.assertIn("Interlocuteur inconnu", self.svc.memoriser_pour(self.autre, "Un fait", "fait"))

    def test_foyer_partage_comme_aria(self):
        tiph = self.personne("Tiphaine")["id"]
        self.svc.memoriser_pour(tiph, "La maison a un jardin", "foyer")
        self.assertEqual(self.rec("jardin")["o"], "")

    def test_recherche_interlocuteur_foyer_et_observations_propres(self):
        tiph = self.personne("Tiphaine")["id"]
        self.svc.memoriser_pour(tiph, "Tiphaine aime le thé vert", "preference")
        self.svc.memoriser(" Batterie à douze volts", "fait", principal=self.bamboo)
        out = self.svc.rechercher_pour(tiph, self.bamboo, "thé")
        self.assertIn("thé vert", out)
        out = self.svc.rechercher_pour(tiph, self.bamboo, "batterie")
        self.assertIn("Tes propres observations", out)
        self.assertIn("Batterie", out)

    def test_recherche_des_observations_ne_ramene_jamais_celles_d_un_autre_acces(self):
        """Les observations propres sont filtrées par propriétaire côté service ET par le Store : on teste les deux sans compter sur l'un."""
        tiph = self.personne("Tiphaine")["id"]
        self.svc.memoriser("Observation secrète de l'autre accès", "fait", principal=self.autre)
        self.svc.memoriser("Observation de Bamboo", "fait", principal=self.bamboo)
        # Le Store filtre déjà par propriétaire ; on neutralise ce filtre SEULEMENT pour la recherche des observations propres, afin de vérifier que
        # le filtre du service (propriétaire = l'accès) tient seul
        vrai = self.s.search

        def sans_filtre(question, limit=5, touch=True, src="llm", owner=None, everything=False):
            return vrai(question, limit, touch, src, owner, everything=True) if owner == self.autre else vrai(question, limit, touch, src, owner, everything)
        self.s.search = sans_filtre
        sortie = self.svc.rechercher_pour(tiph, self.autre, "observation")
        self.assertIn("Tes propres observations", sortie)
        self.assertIn("secrète", sortie)                                       # la sienne
        self.assertNotIn("Observation de Bamboo", sortie.split("Tes propres observations")[1])

    def test_ne_lit_pas_les_souvenirs_prives_d_une_autre_personne(self):
        tiph = self.personne("Tiphaine")["id"]
        out = self.svc.rechercher_pour(tiph, self.bamboo, "science-fiction")
        self.assertNotIn("science-fiction", out)                              # souvenir privé d'Olivier

    def test_observations_d_un_autre_acces_invisibles(self):
        tiph = self.personne("Tiphaine")["id"]
        self.svc.memoriser("Observation secrète de l'autre accès", "fait", principal=self.autre)
        self.assertNotIn("secrète", self.svc.rechercher_pour(tiph, self.bamboo, "observation secrète"))

    def test_observation_privee_au_systeme(self):
        self.svc.memoriser("Le lidar est muet", "fait", principal=self.bamboo)
        self.assertEqual(self.rec("lidar")["o"], self.bamboo)

    def test_oublier_ses_souvenirs_ou_ceux_de_l_interlocuteur_jamais_ceux_d_un_autre(self):
        tiph = self.personne("Tiphaine")["id"]
        self.svc.memoriser_pour(tiph, "Tiphaine aime le thé", "preference")
        self.svc.memoriser("Mesure à oublier", "fait", principal=self.bamboo)
        privé_olivier = self.rec("science-fiction")["id"]
        self.assertEqual(self.svc.oublier_pour(tiph, self.bamboo, self.rec("thé")["id"]), "Oublié.")
        self.assertEqual(self.svc.oublier_pour(tiph, self.bamboo, self.rec("Mesure")["id"]), "Oublié.")
        self.assertEqual(self.svc.oublier_pour(tiph, self.bamboo, privé_olivier), "Souvenir introuvable.")
        self.assertIn(privé_olivier, self.s.records)

    def test_relier_n_importe_quelles_personnes_comme_aria(self):
        tiph = self.personne("Tiphaine")["id"]
        out = self.svc.relier_pour(tiph, "Marion", "enfant", "Tiphaine")
        self.assertTrue(out.startswith("OK"), out)
        marion = self.personne("Marion")
        self.assertIsNotNone(marion)
        self.assertNotIn("a_valider", marion)                                 # fiche créée sans validation (interlocuteur vérifié)
        self.assertEqual(self.rec("Marion")["o"], "")                         # lien public

    def test_relier_avec_moi_est_l_interlocuteur(self):
        tiph = self.personne("Tiphaine")["id"]
        self.svc.relier_pour(tiph, "moi", "ami", "Olivier")
        self.assertIn("Tiphaine est ami(e)", self.rec("Tiphaine est ami")["txt"])

    def test_relier_sans_interlocuteur_refuse(self):
        self.assertIn("Interlocuteur inconnu", self.svc.relier_pour("", "Marion", "enfant", "Tiphaine"))


class IdentificationTests(Base):
    def test_identifier_session_ne_touche_pas_a_aria(self):
        texte, pid = self.i.identifier_session("Paul", trace=None)
        self.assertIn("créé", texte)
        self.assertEqual(self.s.personnes.get(pid)["nom"], "Paul")
        self.assertEqual(self.s.active, self.olivier)

    def test_identifier_session_retrouve_une_personne_connue(self):
        _, pid = self.i.identifier_session("Tiphaine")
        self.assertEqual(pid, self.personne("Tiphaine")["id"])

    def test_homonymes_signales_sans_rien_activer(self):
        self.s.personnes.add("Paul", "Martin")
        self.s.personnes.add("Paul", "Durand")
        texte, pid = self.i.identifier_session("Paul", courant="")
        self.assertIn("Plusieurs personnes", texte)
        self.assertEqual(pid, "")

    def test_une_session_sans_interlocuteur_ne_complete_jamais_la_fiche_d_aria(self):
        """Régression : « pas d'interlocuteur » ('') ne doit pas retomber sur l'interlocuteur actif d'Aria (surnom ajouté à SA fiche)."""
        avant = dict(self.s.personnes.get(self.olivier))
        texte, pid = self.i.identifier_session("Olivier", "Cousinier", "Oli", courant="")
        self.assertEqual(self.s.active, self.olivier)
        self.assertEqual(pid, self.olivier)                                    # c'est bien Olivier qui est désigné pour la SESSION...
        self.assertEqual(self.s.personnes.get(self.olivier)["surnom"], "Oli")  # ... et le surnom est posé par la voie normale (résolution), pas par la fiche active
        self.assertEqual(self.s.personnes.get(self.olivier)["nom"], avant["nom"])

    def test_la_fiche_de_l_interlocuteur_d_aria_n_est_pas_celle_de_la_session(self):
        """Aria parle à Olivier ; la session d'un tiers parle à Tiphaine : un complément donné pour la session va à la fiche de Tiphaine."""
        tiph = self.personne("Tiphaine")["id"]
        self.i.identifier_session("Tiphaine", surnom="Titi", courant=tiph)
        self.assertEqual(self.s.personnes.get(tiph)["surnom"], "Titi")
        self.assertEqual(self.s.personnes.get(self.olivier)["surnom"], "")

    def test_un_acces_ne_peut_pas_etre_designe_comme_interlocuteur(self):
        texte, pid = self.i.identifier_session("Bamboo")
        self.assertEqual(pid, "")
        self.assertIn("accès d'application", texte)

    def test_complete_la_fiche_de_l_interlocuteur_courant(self):
        tiph = self.personne("Tiphaine")["id"]
        texte, pid = self.i.identifier_session("Tiphaine", surnom="Titi", courant=tiph)
        self.assertEqual(pid, tiph)
        self.assertEqual(self.s.personnes.get(tiph)["surnom"], "Titi")

    def test_trace_sur_la_personne_creee(self):
        tr = {"echange": "e_abcdef123456", "ts": "t", "src": "k_bamboo:Bamboo", "who": "Bamboo", "x": "Je m'appelle Paul"}
        _, pid = self.i.identifier_session("Paul", trace=tr)
        self.assertEqual(self.s.personnes.get(pid)["tr"]["x"], "Je m'appelle Paul")


@unittest.skipUnless(HAVE, "mcp / starlette absents (lancer dans le conteneur)")
class CanalTests(Base):
    def setUp(self):
        super().setUp()
        self.app = http_facade.make_app(self.s, self.svc, SECRET)
        self.client = TestClient(self.app.inner_i, base_url="http://127.0.0.1:7872")
        self.client.__enter__()
        self.direct = TestClient(self.app.inner, base_url="http://127.0.0.1:7872")
        self.direct.__enter__()

    def tearDown(self):
        self.direct.__exit__(None, None, None)
        self.client.__exit__(None, None, None)
        super().tearDown()

    def appel(self, outil, args, cle="k_bamboo", nom="Bamboo", session="s1", speaker=None, chemin="/mcp-i", client=None, meta=None):
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json", "X-Memory-Principal": cle,
                   "X-Memory-Name": http_facade.encode_name(nom), "X-Memory-Session": session}
        if speaker:
            headers["X-Memory-Speaker"] = http_facade.encode_speaker(*speaker)
        params = {"name": outil, "arguments": args}
        if meta:
            params["_meta"] = meta
        r = (client or self.client).post(chemin, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}, headers=headers)
        body = r.json()
        return body["result"]["content"][0]["text"] if "result" in body else json.dumps(body)

    def outils(self, chemin="/mcp-i", client=None):
        r = (client or self.client).post(chemin, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                          headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"})
        return sorted(t["name"] for t in r.json()["result"]["tools"])

    def test_outils_du_canal_interlocuteur(self):
        self.assertEqual(self.outils(), ["identifier_interlocuteur", "memoriser", "noter_observation", "oublier", "relier", "rechercher_souvenir"][:0]
                         or sorted(["identifier_interlocuteur", "memoriser", "noter_observation", "oublier", "relier", "rechercher_souvenir"]))

    def test_le_canal_d_origine_ne_change_pas(self):
        self.assertEqual(self.outils("/mcp", self.direct), ["memoriser", "oublier", "rechercher_souvenir", "relier"])

    def test_interlocuteur_fourni_par_l_application(self):
        out = self.appel("memoriser", {"fait": "Tiphaine aime le thé", "type": "preference"}, speaker=("Tiphaine", "Cousinier"))
        self.assertTrue(out.startswith("OK"), out)
        self.assertEqual(self.rec("thé")["o"], self.personne("Tiphaine")["id"])
        self.assertEqual(self.s.active, self.olivier)

    def test_l_interlocuteur_fourni_est_retenu_pour_la_session(self):
        self.appel("memoriser", {"fait": "Premier fait"}, speaker=("Tiphaine", "Cousinier"))
        out = self.appel("memoriser", {"fait": "Deuxième fait"}, speaker=None)           # l'application ne le renvoie pas à chaque appel
        self.assertTrue(out.startswith("OK"), out)
        self.assertEqual(self.rec("Deuxième")["o"], self.personne("Tiphaine")["id"])

    def test_sans_interlocuteur_le_llm_doit_l_identifier(self):
        self.assertIn("Interlocuteur inconnu", self.appel("memoriser", {"fait": "Un fait"}))
        self.assertIsNone(self.rec("Un fait"))
        self.assertIn("créé", self.appel("identifier_interlocuteur", {"nom": "Paul"}))
        self.assertTrue(self.appel("memoriser", {"fait": "Paul aime le vélo"}).startswith("OK"))
        self.assertEqual(self.rec("vélo")["o"], self.personne("Paul")["id"])
        self.assertEqual(self.s.active, self.olivier)

    def test_si_l_application_a_identifie_le_llm_ne_peut_pas_changer_d_interlocuteur(self):
        self.appel("memoriser", {"fait": "Fait de Tiphaine"}, speaker=("Tiphaine", "Cousinier"))
        out = self.appel("identifier_interlocuteur", {"nom": "Olivier"}, speaker=("Tiphaine", "Cousinier"))
        self.assertIn("déjà identifié", out)
        self.appel("memoriser", {"fait": "Encore un fait"}, speaker=("Tiphaine", "Cousinier"))
        self.assertEqual(self.rec("Encore un fait")["o"], self.personne("Tiphaine")["id"])

    def test_deux_sessions_deux_interlocuteurs_sans_melange(self):
        self.appel("memoriser", {"fait": "Fait de Tiphaine"}, session="sA", speaker=("Tiphaine", "Cousinier"))
        self.appel("identifier_interlocuteur", {"nom": "Paul"}, session="sB")
        self.appel("memoriser", {"fait": "Fait de Paul"}, session="sB")
        self.assertEqual(self.rec("Fait de Tiphaine")["o"], self.personne("Tiphaine")["id"])
        self.assertEqual(self.rec("Fait de Paul")["o"], self.personne("Paul")["id"])

    def test_deux_acces_meme_nom_de_session_sont_separes(self):
        self.appel("identifier_interlocuteur", {"nom": "Paul"}, cle="k_bamboo", session="s1")
        self.assertIn("Interlocuteur inconnu", self.appel("memoriser", {"fait": "Fait de l'autre accès"}, cle="k_autre", nom="Autre", session="s1"))

    def test_observation_privee_au_systeme_et_interlocuteur_intact(self):
        self.appel("memoriser", {"fait": "Fait de Tiphaine"}, speaker=("Tiphaine", "Cousinier"))
        self.appel("noter_observation", {"fait": "Le lidar est muet"}, speaker=("Tiphaine", "Cousinier"))
        self.assertEqual(self.rec("lidar")["o"], self.bamboo)
        self.assertEqual(self.rec("Fait de Tiphaine")["o"], self.personne("Tiphaine")["id"])

    def test_relier_n_importe_quelles_personnes(self):
        out = self.appel("relier", {"personne_a": "Marion", "lien": "enfant", "personne_b": "Tiphaine"}, speaker=("Tiphaine", "Cousinier"))
        self.assertTrue(out.startswith("OK"), out)
        self.assertEqual(self.rec("Marion")["o"], "")

    def test_la_trace_d_audit_suit_les_ecritures(self):
        meta = {"echange": "e_abcdef123456", "ts": "t", "src": "k_bamboo:Bamboo", "who": "Tiphaine", "x": "Tiphaine aime le thé"}
        self.appel("memoriser", {"fait": "Tiphaine aime le thé"}, speaker=("Tiphaine", "Cousinier"), meta=meta)
        self.assertEqual(self.rec("thé")["tr"]["who"], "Tiphaine")

    def test_un_argument_du_llm_ne_peut_pas_designer_l_interlocuteur(self):
        out = self.appel("memoriser", {"fait": "Fait forgé", "speaker": self.olivier, "o": self.olivier, "owner": self.olivier})
        self.assertIn("Interlocuteur inconnu", out)
        self.assertIsNone(self.rec("Fait forgé"))

    def test_authentification_du_canal_inchangee(self):
        """Le garde (secret) s'applique aussi à /mcp-i : testé sur l'application complète (le garde répond avant tout outil, sans gestionnaire de session)."""
        import asyncio
        reponses = []

        async def appeler(autorisation):
            messages = []
            headers = [(b"content-type", b"application/json"), (b"host", b"127.0.0.1:7872")]
            if autorisation:
                headers.append((b"authorization", autorisation))

            async def receive():
                return {"type": "http.request", "body": b"{}", "more_body": False}

            async def send(message):
                messages.append(message)
            scope = {"type": "http", "method": "POST", "path": "/mcp-i", "headers": headers, "query_string": b"", "http_version": "1.1", "scheme": "http"}
            await self.app(scope, receive, send)
            return messages[0]["status"]
        for autorisation in (None, b"Bearer faux", b"Basic " + SECRET.encode()):
            reponses.append(asyncio.run(appeler(autorisation)))
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.assertEqual(reponses, [401, 401, 401])

    def test_concurrence_chaque_session_son_interlocuteur(self):
        noms = ["Anne", "Bruno", "Chloé", "David", "Émile", "Fanny"]
        erreurs = []

        def tour(n, nom):
            try:
                self.appel("identifier_interlocuteur", {"nom": nom}, session=f"c{n}", cle=f"k_c{n}", nom=f"Acces{n}")
                self.appel("memoriser", {"fait": f"Fait de {nom} numéro {n}"}, session=f"c{n}", cle=f"k_c{n}", nom=f"Acces{n}")
            except Exception as e:                                             # pragma: no cover
                erreurs.append(e)
        ts = [threading.Thread(target=tour, args=(n, nom)) for n, nom in enumerate(noms)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(erreurs, [])
        for n, nom in enumerate(noms):
            self.assertEqual(self.rec(f"Fait de {nom} numéro {n}")["o"], self.personne(nom)["id"], nom)


if __name__ == "__main__":
    unittest.main()
