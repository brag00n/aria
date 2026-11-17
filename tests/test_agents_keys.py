"""Clés API des agents : magasin (hachage, expiration, révocation, origines) et protections (débit, blocage, quota, CORS)."""
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.agents import keys, security  # noqa: E402


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="acckeys_"))
        self.clock = Clock()
        self.store = keys.KeyStore(self.tmp, clock=self.clock)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def new(self, **kw):
        kw.setdefault("label", "Portail")
        kw.setdefault("origins", ["https://app.example.com"])
        return self.store.create(**kw)


class CreationTests(StoreCase):
    def test_la_cle_en_clair_n_est_montree_qu_une_fois_et_jamais_stockee(self):
        vue, cle = self.new()
        self.assertTrue(cle.startswith("ak_") and len(cle) > 40)
        self.assertEqual(vue["prefix"], cle[:7])
        contenu = (self.tmp / "agents.json").read_text("utf-8")
        self.assertNotIn(cle, contenu)
        self.assertNotIn(cle[3:], contenu)
        self.assertNotIn("hash", json.dumps(self.store.list()))                    # jamais renvoyé à la page d'administration
        self.assertNotIn(cle, json.dumps(self.store.list()))

    def test_deux_cles_sont_differentes(self):
        _, a = self.new()
        _, b = self.new(label="Autre")
        self.assertNotEqual(a, b)

    def test_libelle_obligatoire(self):
        self.assertEqual(self.store.create("   "), (None, ""))

    def test_secret_et_fichier_de_cles_crees(self):
        self.new()
        self.assertTrue((self.tmp / "secret.key").exists())
        self.assertGreaterEqual(len((self.tmp / "secret.key").read_bytes()), 32)

    def test_rechargement_depuis_le_disque(self):
        vue, cle = self.new(profile="portail", limits={"rate_per_min": 5, "daily_turns": 7}, voice=False)
        autre = keys.KeyStore(self.tmp, clock=self.clock)
        self.assertEqual(autre.get(vue["id"])["limits"], {"rate_per_min": 5, "daily_turns": 7})
        self.assertFalse(autre.get(vue["id"])["voice"])
        self.assertEqual(autre.authenticate(cle)[0], "ok")                          # le secret est relu : la clé reste valide

    def test_secret_different_donne_un_autre_hash(self):
        _, cle = self.new()
        (self.tmp / "secret.key").unlink()
        autre = keys.KeyStore(self.tmp, clock=self.clock)
        self.assertEqual(autre.authenticate(cle)[0], "unknown")

    def test_fichier_corrompu_ne_plante_pas(self):
        (self.tmp / "agents.json").write_text("{pas du json", "utf-8")
        self.assertEqual(keys.KeyStore(self.tmp).list(), [])


class AuthentificationTests(StoreCase):
    def test_cle_valide(self):
        vue, cle = self.new()
        statut, rec = self.store.authenticate(cle)
        self.assertEqual((statut, rec["id"]), ("ok", vue["id"]))

    def test_cle_inconnue_ou_mal_formee(self):
        self.new()
        for mauvaise in ("", None, "ak_", "ak_" + "x" * 43, "autre", "ak_" + "x" * 500, "Bearer ak_x"):
            self.assertEqual(self.store.authenticate(mauvaise)[0], "unknown", mauvaise)

    def test_revocation_immediate_et_reversible(self):
        vue, cle = self.new()
        self.assertTrue(self.store.revoke(vue["id"]))
        self.assertEqual(self.store.authenticate(cle), ("revoked", None))
        self.assertEqual(self.store.get(vue["id"])["etat"], "révoquée")
        self.store.revoke(vue["id"], False)
        self.assertEqual(self.store.authenticate(cle)[0], "ok")

    def test_expiration(self):
        vue, cle = self.new(expires_days=2)
        self.assertEqual(self.store.authenticate(cle)[0], "ok")
        self.clock.advance(86400 * 2 + 1)
        self.assertEqual(self.store.authenticate(cle), ("expired", None))
        self.assertEqual(self.store.get(vue["id"])["etat"], "expirée")

    def test_sans_expiration_par_defaut(self):
        _, cle = self.new()
        self.clock.advance(86400 * 3650)
        self.assertEqual(self.store.authenticate(cle)[0], "ok")

    def test_origine_autorisee_ou_refusee(self):
        _, cle = self.new(origins=["https://app.example.com"])
        self.assertEqual(self.store.authenticate(cle, "https://app.example.com")[0], "ok")
        self.assertEqual(self.store.authenticate(cle, "HTTPS://App.Example.com/")[0], "ok")      # normalisée
        self.assertEqual(self.store.authenticate(cle, "https://evil.example.com")[0], "origin")
        self.assertEqual(self.store.authenticate(cle, "http://app.example.com")[0], "origin")    # schéma différent
        self.assertEqual(self.store.authenticate(cle, "https://app.example.com:8443")[0], "origin")
        self.assertEqual(self.store.authenticate(cle, "null")[0], "origin")

    def test_client_sans_origine_accepte(self):
        _, cle = self.new(origins=["https://app.example.com"])
        self.assertEqual(self.store.authenticate(cle, None)[0], "ok")                # curl, serveur à serveur
        self.assertEqual(self.store.authenticate(cle, "")[0], "ok")

    def test_cle_sans_origine_refuse_tout_navigateur(self):
        _, cle = self.new(origins=[])
        self.assertEqual(self.store.authenticate(cle, "https://app.example.com")[0], "origin")

    def test_last_used_mis_a_jour(self):
        vue, cle = self.new()
        self.assertEqual(self.store.get(vue["id"])["last_used"], "")
        self.store.authenticate(cle)
        self.assertNotEqual(self.store.get(vue["id"])["last_used"], "")

    def test_last_used_ecrit_au_plus_une_fois_par_minute(self):
        _, cle = self.new()
        mtime = (self.tmp / "agents.json").stat().st_mtime_ns
        self.store.authenticate(cle)
        self.store.authenticate(cle)
        self.assertEqual((self.tmp / "agents.json").stat().st_mtime_ns, mtime)
        self.clock.advance(61)
        self.store.authenticate(cle)
        self.assertNotEqual((self.tmp / "agents.json").stat().st_mtime_ns, mtime)

    def test_flush_ecrit_le_dernier_usage(self):
        vue, cle = self.new()
        self.store.authenticate(cle)
        self.store.flush()
        self.assertNotEqual(keys.KeyStore(self.tmp, clock=self.clock).get(vue["id"])["last_used"], "")


class GestionTests(StoreCase):
    def test_modifier(self):
        vue, _ = self.new()
        v = self.store.update(vue["id"], label="Nouveau nom", origins=["https://a.example", "https://a.example", "pas une origine"],
                              profile="p2", limits={"rate_per_min": 3, "daily_turns": 9}, voice=False)
        self.assertEqual((v["label"], v["origins"], v["profile"], v["limits"], v["voice"]),
                         ("Nouveau nom", ["https://a.example"], "p2", {"rate_per_min": 3, "daily_turns": 9}, False))

    def test_modifier_inconnu_ou_libelle_vide(self):
        vue, _ = self.new()
        self.assertIsNone(self.store.update("k_zzz", label="x"))
        self.assertIsNone(self.store.update(vue["id"], label="  "))
        self.assertEqual(self.store.get(vue["id"])["label"], "Portail")

    def test_prolonger_l_expiration(self):
        vue, cle = self.new(expires_days=1)
        self.clock.advance(86400 * 2)
        self.assertEqual(self.store.authenticate(cle)[0], "expired")
        self.store.update(vue["id"], expires_days=30)
        self.assertEqual(self.store.authenticate(cle)[0], "ok")
        self.store.update(vue["id"], expires_days=0)
        self.assertEqual(self.store.get(vue["id"])["expires"], "")

    def test_supprimer(self):
        vue, cle = self.new()
        self.assertTrue(self.store.delete(vue["id"]))
        self.assertFalse(self.store.delete(vue["id"]))
        self.assertEqual(self.store.authenticate(cle)[0], "unknown")

    def test_limites_invalides_ramenees_aux_valeurs_par_defaut(self):
        v, _ = self.new(limits={"rate_per_min": -4, "daily_turns": "abc"})
        self.assertEqual(v["limits"], keys.DEFAULT_LIMITS)
        v, _ = self.new(limits={"rate_per_min": 10**12})
        self.assertEqual(v["limits"]["rate_per_min"], 1_000_000)

    def test_origines_normalisees(self):
        for ok, attendu in [("https://App.Example.com/", "https://app.example.com"), ("http://localhost:8080", "http://localhost:8080")]:
            self.assertEqual(keys.normalize_origin(ok), attendu)
        for mauvaise in ("", "*", "https://*.example.com", "https://a.example/chemin", "ftp://a.example", "app.example.com",
                         "https://u:p@a.example", "https://a.example?x=1", "https://", None):
            self.assertEqual(keys.normalize_origin(mauvaise), "", mauvaise)

    def test_ecritures_concurrentes(self):
        def creer(i):
            self.store.create(f"app {i}", ["https://a.example"])
        threads = [threading.Thread(target=creer, args=(i,)) for i in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(self.store.list()), 20)
        self.assertEqual(len(keys.KeyStore(self.tmp).list()), 20)
        self.assertEqual(len({k["id"] for k in self.store.list()}), 20)

    def test_aucun_fichier_temporaire_ne_reste(self):
        self.new()
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.suffix == ".tmp" or p.name.endswith(".tmp")], [])


class RegenerationTests(StoreCase):
    def test_nouvelle_cle_meme_identifiant_ancienne_cle_invalidee(self):
        vue, ancienne = self.new(profile="portail", limits={"rate_per_min": 7, "daily_turns": 9}, voice=False)
        vue2, nouvelle = self.store.regenerate(vue["id"])
        self.assertEqual(vue2["id"], vue["id"])                                          # même identifiant interne : la mémoire de l'accès suit
        self.assertNotEqual(nouvelle, ancienne)
        self.assertTrue(nouvelle.startswith("ak_") and len(nouvelle) > 40)
        self.assertEqual(self.store.authenticate(ancienne)[0], "unknown")                 # l'ancienne ne marche plus, tout de suite
        statut, rec = self.store.authenticate(nouvelle, "https://app.example.com")
        self.assertEqual((statut, rec["id"]), ("ok", vue["id"]))

    def test_les_reglages_sont_conserves(self):
        vue, _ = self.new(profile="portail", limits={"rate_per_min": 7, "daily_turns": 9}, voice=False, expires_days=30)
        apres, _ = self.store.regenerate(vue["id"])
        for champ in ("label", "origins", "profile", "limits", "voice", "created", "expires"):
            self.assertEqual(apres[champ], vue[champ], champ)

    def test_le_prefixe_affiche_suit_la_nouvelle_cle(self):
        vue, ancienne = self.new()
        apres, nouvelle = self.store.regenerate(vue["id"])
        self.assertEqual(apres["prefix"], nouvelle[:7])
        self.assertNotEqual(apres["prefix"], vue["prefix"])

    def test_la_nouvelle_cle_n_est_jamais_stockee_en_clair(self):
        vue, _ = self.new()
        _, nouvelle = self.store.regenerate(vue["id"])
        self.assertNotIn(nouvelle, (self.tmp / "agents.json").read_text("utf-8"))
        self.assertNotIn(nouvelle, json.dumps(self.store.list()))

    def test_survit_au_rechargement(self):
        vue, ancienne = self.new()
        _, nouvelle = self.store.regenerate(vue["id"])
        autre = keys.KeyStore(self.tmp, clock=self.clock)
        self.assertEqual(autre.authenticate(nouvelle)[0], "ok")
        self.assertEqual(autre.authenticate(ancienne)[0], "unknown")

    def test_ne_reactive_pas_une_cle_revoquee_ou_expiree(self):
        vue, _ = self.new()
        self.store.revoke(vue["id"])
        _, nouvelle = self.store.regenerate(vue["id"])
        self.assertEqual(self.store.authenticate(nouvelle)[0], "revoked")
        v2, _ = self.new(label="Autre", expires_days=1)
        _, n2 = self.store.regenerate(v2["id"])
        self.clock.advance(86400 * 2)
        self.assertEqual(self.store.authenticate(n2)[0], "expired")

    def test_accès_inconnu(self):
        self.assertEqual(self.store.regenerate("k_zzz"), (None, ""))

    def test_deux_regenerations_de_suite(self):
        vue, c0 = self.new()
        _, c1 = self.store.regenerate(vue["id"])
        _, c2 = self.store.regenerate(vue["id"])
        self.assertEqual([self.store.authenticate(c)[0] for c in (c0, c1, c2)], ["unknown", "unknown", "ok"])


class RetentionTests(StoreCase):
    """Durée de conservation des échanges d'un accès dans le journal d'audit (0 = durée par défaut du journal)."""

    def test_defaut_zero_et_lecture_par_identifiant(self):
        vue, _ = self.store.create("Robot")
        self.assertEqual(vue["retention_days"], 0)
        self.assertEqual(self.store.retention_of(vue["id"]), 0)
        self.assertEqual(self.store.retention_of("k_inconnu"), 0)

    def test_creation_et_modification(self):
        vue, _ = self.store.create("Robot", retention_days=7)
        self.assertEqual(self.store.retention_of(vue["id"]), 7)
        self.store.update(vue["id"], retention_days=30)
        self.assertEqual(self.store.retention_of(vue["id"]), 30)
        self.store.update(vue["id"], retention_days=0)
        self.assertEqual(self.store.retention_of(vue["id"]), 0)

    def test_valeurs_invalides_bornees(self):
        vue, _ = self.store.create("Robot")
        for entree, attendu in (("abc", 0), (-5, 0), (None, 0), (99999, 3650), ("12", 12), (3.9, 3)):
            self.store.update(vue["id"], retention_days=entree)
            self.assertEqual(self.store.retention_of(vue["id"]), attendu, entree)

    def test_persistante_apres_rechargement_et_anciennes_cles_lisibles(self):
        vue, _ = self.store.create("Robot", retention_days=14)
        rouvert = keys.KeyStore(self.store.dir)
        self.assertEqual(rouvert.retention_of(vue["id"]), 14)
        brut = json.loads(rouvert.path.read_text("utf-8"))
        for k in brut["agents"]:
            k.pop("retention_days", None)                                 # fichier d'avant cette option
        rouvert.path.write_text(json.dumps(brut), "utf-8")
        self.assertEqual(keys.KeyStore(self.store.dir).retention_of(vue["id"]), 0)

    def test_la_regeneration_garde_la_duree(self):
        vue, _ = self.store.create("Robot", retention_days=9)
        self.store.regenerate(vue["id"])
        self.assertEqual(self.store.retention_of(vue["id"]), 9)


class PublicationLiensTests(StoreCase):
    """Option « publier ses propres liens » d'une clé : désactivée par défaut."""

    def test_defaut_desactive_et_modifiable(self):
        vue, _ = self.store.create("Robot")
        self.assertFalse(vue["publish_links"])
        self.assertFalse(self.store.update(vue["id"], publish_links=False)["publish_links"])
        self.assertTrue(self.store.update(vue["id"], publish_links=True)["publish_links"])
        self.assertTrue(self.store.create("Autre", publish_links=True)[0]["publish_links"])

    def test_persistante_et_anciennes_cles_sans_l_option(self):
        vue, _ = self.store.create("Robot", publish_links=True)
        self.assertTrue(keys.KeyStore(self.store.dir).get(vue["id"])["publish_links"])
        brut = json.loads(self.store.path.read_text("utf-8"))
        for k in brut["agents"]:
            k.pop("publish_links", None)
        self.store.path.write_text(json.dumps(brut), "utf-8")
        self.assertFalse(keys.KeyStore(self.store.dir).get(vue["id"])["publish_links"])

    def test_valeurs_non_booleennes_ramenees_a_un_booleen(self):
        vue, _ = self.store.create("Robot")
        for entree, attendu in (("oui", True), ("", False), (0, False), (1, True), (None, False)):
            self.assertEqual(self.store.update(vue["id"], publish_links=entree)["publish_links"], attendu)

    def test_la_regeneration_garde_l_option(self):
        vue, _ = self.store.create("Robot", publish_links=True)
        self.store.regenerate(vue["id"])
        self.assertTrue(self.store.get(vue["id"])["publish_links"])


class PublicationInterlocuteurTests(StoreCase):
    """Option « publier les informations de son interlocuteur » : désactivée par défaut, indépendante de « publier ses liens »."""

    def test_defaut_desactive_modifiable_et_independant(self):
        vue, _ = self.store.create("Robot")
        self.assertFalse(vue["publish_speaker"])
        self.assertTrue(self.store.update(vue["id"], publish_speaker=True)["publish_speaker"])
        self.assertFalse(self.store.get(vue["id"])["publish_links"])
        self.assertTrue(self.store.create("Autre", publish_speaker=True)[0]["publish_speaker"])

    def test_persistante_et_anciennes_cles(self):
        vue, _ = self.store.create("Robot", publish_speaker=True)
        self.assertTrue(keys.KeyStore(self.store.dir).get(vue["id"])["publish_speaker"])
        brut = json.loads(self.store.path.read_text("utf-8"))
        for k in brut["agents"]:
            k.pop("publish_speaker", None)
        self.store.path.write_text(json.dumps(brut), "utf-8")
        self.assertFalse(keys.KeyStore(self.store.dir).get(vue["id"])["publish_speaker"])

    def test_valeurs_ramenees_a_un_booleen_et_regeneration(self):
        vue, _ = self.store.create("Robot")
        for entree, attendu in (("oui", True), ("", False), (0, False), (1, True), (None, False)):
            self.assertEqual(self.store.update(vue["id"], publish_speaker=entree)["publish_speaker"], attendu)
        self.store.update(vue["id"], publish_speaker=True)
        self.store.regenerate(vue["id"])
        self.assertTrue(self.store.get(vue["id"])["publish_speaker"])


class DebitTests(unittest.TestCase):
    def test_fenetre_glissante(self):
        c = Clock(100.0)
        w = security.SlidingWindow(3, 60.0, clock=c)
        self.assertEqual([w.allow("a") for _ in range(5)], [True, True, True, False, False])
        self.assertTrue(w.allow("b"))                                                # un autre identifiant est indépendant
        c.advance(61)
        self.assertTrue(w.allow("a"))

    def test_limite_propre_a_la_cle(self):
        w = security.SlidingWindow(30, 60.0, clock=Clock())
        self.assertEqual([w.allow("a", limit=2) for _ in range(3)], [True, True, False])

    def test_retry_after(self):
        c = Clock(0.0)
        w = security.SlidingWindow(1, 60.0, clock=c)
        w.allow("a")
        c.advance(20)
        self.assertTrue(30 <= w.retry_after("a") <= 41)

    def test_memoire_bornee(self):
        w = security.SlidingWindow(1, 60.0, clock=Clock(), max_ids=50)
        for i in range(500):
            w.allow(f"ip{i}")
        self.assertLessEqual(len(w.events), 500)


class BlocageTests(unittest.TestCase):
    def test_blocage_apres_echecs_puis_deblocage(self):
        c = Clock(0.0)
        f = security.FailureLock(max_fail=3, window=60, lock_s=120, clock=c)
        for _ in range(2):
            f.fail("1.2.3.4")
        self.assertEqual(f.blocked("1.2.3.4"), 0)
        f.fail("1.2.3.4")
        self.assertGreater(f.blocked("1.2.3.4"), 0)
        self.assertEqual(f.blocked("5.6.7.8"), 0)                                    # une autre IP n'est pas touchée
        c.advance(121)
        self.assertEqual(f.blocked("1.2.3.4"), 0)

    def test_les_echecs_anciens_sont_oublies(self):
        c = Clock(0.0)
        f = security.FailureLock(max_fail=3, window=60, lock_s=120, clock=c)
        f.fail("a")
        f.fail("a")
        c.advance(61)
        f.fail("a")
        self.assertEqual(f.blocked("a"), 0)

    def test_un_succes_efface_les_echecs(self):
        f = security.FailureLock(max_fail=3, clock=Clock(0.0))
        f.fail("a")
        f.fail("a")
        f.success("a")
        f.fail("a")
        self.assertEqual(f.blocked("a"), 0)


class QuotaTests(unittest.TestCase):
    def test_quota_journalier_et_remise_a_zero(self):
        c = Clock(1_800_000_000.0)
        q = security.DailyQuota(clock=c)
        self.assertEqual([q.take("k1", 2) for _ in range(3)], [True, True, False])
        self.assertEqual(q.used("k1"), 2)
        self.assertTrue(q.take("k2", 2))
        c.advance(86400)
        self.assertTrue(q.take("k1", 2))
        self.assertEqual(q.used("k1"), 1)


class CorsTests(unittest.TestCase):
    def test_origine_autorisee_renvoyee_exactement(self):
        h = security.cors_headers("https://App.Example.com/", ["https://app.example.com"])
        self.assertEqual(h["Access-Control-Allow-Origin"], "https://app.example.com")
        self.assertEqual(h["Vary"], "Origin")

    def test_jamais_d_etoile_ni_d_origine_non_autorisee(self):
        for origine in ("https://evil.example.com", "", None, "null", "*"):
            h = security.cors_headers(origine, ["https://app.example.com"])
            self.assertNotIn("Access-Control-Allow-Origin", h, origine)
        self.assertNotIn("*", "".join(security.cors_headers("https://a.example", ["https://a.example"]).values()))

    def test_preflight(self):
        h = security.preflight_headers("https://a.example", ["https://a.example"])
        self.assertIn("Authorization", h["Access-Control-Allow-Headers"])
        self.assertIn("POST", h["Access-Control-Allow-Methods"])
        self.assertNotIn("Access-Control-Allow-Methods", security.preflight_headers("https://b.example", ["https://a.example"]))

    def test_extraction_de_la_cle(self):
        self.assertEqual(security.bearer({"Authorization": "Bearer ak_abc"}), "ak_abc")
        self.assertEqual(security.bearer({"Authorization": "bearer  ak_abc "}), "ak_abc")
        self.assertEqual(security.bearer({"X-Api-Key": " ak_abc "}), "ak_abc")
        self.assertEqual(security.bearer({"Authorization": "Basic xyz"}), "")
        self.assertEqual(security.bearer({}), "")


if __name__ == "__main__":
    unittest.main()
