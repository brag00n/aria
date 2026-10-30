"""Tests du serveur de mémoire (bibliothèque standard : python -m unittest discover tools/memory/tests)."""
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import scoring  # noqa: E402
import store as store_mod  # noqa: E402
import textproc  # noqa: E402
from activity import Activity  # noqa: E402

DAY = 86400.0


class Clock:
    """Horloge injectable : les tests de décroissance n'attendent jamais."""
    def __init__(self, start=1_800_000_000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, days):
        self.t += days * DAY


class TextprocTests(unittest.TestCase):
    def test_accents_et_casse(self):
        self.assertEqual(textproc.normalize("Éléphant À Noël"), "elephant a noel")

    def test_apostrophe_typographique_separe_comme_l_apostrophe_droite(self):
        self.assertEqual(textproc.analyze("l’école de Léa"), textproc.analyze("l'école de Léa"))
        self.assertEqual(textproc.analyze("l’école de Léa"), ["ecol", "lea"])

    def test_elision_et_mots_vides(self):
        self.assertEqual(textproc.analyze("l'école de Léa"), ["ecol", "lea"])

    def test_pluriels_et_feminins_donnent_la_meme_racine(self):
        self.assertEqual(textproc.analyze("préfère"), textproc.analyze("préférée"))
        self.assertEqual(textproc.analyze("volet"), textproc.analyze("volets"))

    def test_chiffres_conserves(self):
        self.assertIn("19", textproc.analyze("19 degrés"))

    def test_idempotence_sur_la_requete(self):
        a = textproc.analyze("Quelle température préfère Marion ?")
        self.assertEqual(a, textproc.analyze("quelle temperature prefere marion"))


class ScoringTests(unittest.TestCase):
    def test_demi_vie(self):
        # éphémère de note 2 : après une demi-vie (10 jours), 1,0
        self.assertAlmostEqual(scoring.effective(2, 1.0, 0, 10 * DAY, "ephemere", []), 1.0, places=6)

    def test_protege_ne_decroit_pas(self):
        self.assertEqual(scoring.effective(5, 1.0, 0, 10_000 * DAY, "fait", ["pinned"]), 5.0)

    def test_renfort_multiplie(self):
        self.assertEqual(scoring.effective(5, 2.0, 0, 0, "fait", []), 10.0)

    def test_classement_privilegie_importance(self):
        self.assertGreater(scoring.final_score(1.0, 9, False), scoring.final_score(1.0, 1, False))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="memtest_")
        self.clock = Clock()
        self.dirs = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, speaker="p0", **kw):
        """Store de test. Par défaut un interlocuteur est actif (sans cela, plus rien ne se mémorise) ; speaker="" = Inconnue."""
        kw.setdefault("clock", self.clock)
        kw.setdefault("lock_instance", False)
        s = store_mod.Store(self.dirs, "default", **kw)
        s.active = speaker
        return s


class StoreTests(StoreCase):
    def test_ajout_et_note_de_depart_par_type(self):
        s = self.make()
        status, rec = s.add("Marion préfère 19 degrés", "préférence")
        self.assertEqual(status, "added")
        self.assertEqual(rec["t"], "preference")
        self.assertEqual(rec["imp"], 6.0)
        self.assertEqual(s.add("pluie demain", "éphémère")[1]["imp"], 2.0)
        self.assertEqual(s.add("acheter du pain", "tâche")[1]["imp"], 4.0)
        self.assertEqual(s.add("Olivier habite à Lyon")[1]["imp"], 5.0)

    def test_doublon_renforce_au_lieu_de_creer(self):
        s = self.make()
        _, first = s.add("Olivier habite à Lyon")
        status, again = s.add("olivier  HABITE à lyon !")
        self.assertEqual(status, "duplicate")
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(len(s.records), 1)
        self.assertAlmostEqual(first["s"], 1.25)

    def test_plafond_du_renfort(self):
        s = self.make()
        _, rec = s.add("Olivier habite à Lyon")
        for _ in range(20):
            s.add("Olivier habite à Lyon")
        self.assertEqual(rec["s"], scoring.S_MAX)

    def test_recherche_trouve_malgre_accents_et_pluriels(self):
        s = self.make()
        s.add("Marion préfère 19 degrés dans sa chambre", "préférence")
        s.add("Le compteur Linky est l'équipement 270")
        res, fallback = s.search("quelle température préfère Marion ?")
        self.assertFalse(fallback)
        self.assertIn("Marion", res[0]["txt"])

    def _four_memories(self):
        s = self.make()
        for txt, t in [("La chatte s'appelle Réglisse.", "fait"),
                       ("Marion, la fille, préfère 19 degrés dans sa chambre.", "préférence"),
                       ("Le compteur Linky de la maison est l'équipement 270.", "fait"),
                       ("Le code du portail est dans le garage.", "fait")]:
            s.add(txt, t, src="ui")
        return s

    def test_precision_la_femme_ne_renvoie_pas_la_chatte(self):
        """Cas réel : « le nom de ma femme » renvoyait le souvenir de la chatte (mots génériques en commun)."""
        s = self._four_memories()
        for question in ("nom de la femme de l'utilisateur", "quel est le nom de ma femme", "qui est ma voisine"):
            res, fallback = s.search(question, touch=False)
            self.assertTrue(fallback, question)          # signalé comme « pas de correspondance exacte »

    def test_precision_les_bonnes_questions_restent_trouvees(self):
        s = self._four_memories()
        cases = {"comment s'appelle ma chatte": "chatte", "température préférée de Marion": "Marion",
                 "équipement du compteur": "compteur", "où est le code du portail": "portail"}
        for question, word in cases.items():
            res, fallback = s.search(question, touch=False)
            self.assertFalse(fallback, question)
            self.assertIn(word, res[0]["txt"], question)

    def test_recherche_vide_et_sans_resultat_repli(self):
        s = self.make()
        s.add("Olivier habite à Lyon")
        res, fallback = s.search("")
        self.assertEqual(len(res), 1)
        res, fallback = s.search("zzzzqqq")
        self.assertTrue(fallback)
        self.assertEqual(len(res), 1)

    def test_rappel_renforce_et_ecrit_un_touch(self):
        s = self.make()
        _, rec = s.add("Marion préfère 19 degrés", "préférence")
        s.search("Marion")
        self.assertEqual(rec["hits"], 1)
        self.assertGreater(rec["s"], 1.0)
        ops = [json.loads(l)["op"] for l in (s.ns_dir / "events.jsonl").read_text("utf-8").splitlines()]
        self.assertEqual(ops, ["add", "touch"])

    def test_decroissance_masque_puis_archive(self):
        s = self.make()
        _, rec = s.add("il pleut aujourd'hui", "éphémère")
        self.assertEqual(s.state(rec), "actif")
        self.clock.advance(11)
        self.assertEqual(s.state(rec), "masque")
        res, fallback = s.search("pleut")
        self.assertEqual(res, [])          # masqué : ni trouvé ni proposé en repli
        self.clock.advance(60)
        out = s.compact()
        self.assertEqual(out["archived"], 1)
        self.assertEqual(len(s.records), 0)
        archived = s.archived()
        self.assertEqual(archived[0]["why"], "decay")
        self.assertEqual(archived[0]["txt"], "il pleut aujourd'hui")

    def test_tag_protege_ne_decroit_pas(self):
        s = self.make()
        _, rec = s.add("Le code du portail est secret", "éphémère", "pinned")
        self.clock.advance(1000)
        self.assertEqual(s.state(rec), "protege")
        s.compact()
        self.assertIn(rec["id"], s.records)

    def test_oubli_archive_et_restauration(self):
        s = self.make()
        _, rec = s.add("Olivier habite à Lyon")
        self.assertTrue(s.delete(rec["id"]))
        self.assertEqual(len(s.records), 0)
        self.assertEqual(s.archived()[0]["why"], "user")
        self.assertFalse(s.delete(rec["id"]))
        self.assertTrue(s.restore(rec["id"]))
        self.assertIn(rec["id"], s.records)

    def test_rejeu_donne_le_meme_etat(self):
        s = self.make()
        _, a = s.add("Marion préfère 19 degrés", "préférence", "famille")
        _, b = s.add("acheter du pain", "tâche")
        s.search("Marion")
        s.update(b["id"], {"txt": "acheter du pain complet"})
        s.close()
        again = self.make()
        self.assertEqual(set(again.records), {a["id"], b["id"]})
        self.assertEqual(again.records[a["id"]]["hits"], 1)
        self.assertEqual(again.records[b["id"]]["txt"], "acheter du pain complet")
        res, _ = again.search("pain complet")
        self.assertEqual(res[0]["id"], b["id"])

    def test_compaction_idempotente(self):
        s = self.make()
        for i in range(5):
            s.add(f"souvenir numero {i} sur la maison")
        s.compact()
        first = (s.ns_dir / "memories.jsonl").read_bytes()
        s.compact()
        self.assertEqual((s.ns_dir / "memories.jsonl").read_bytes(), first)
        self.assertEqual((s.ns_dir / "events.jsonl").read_bytes(), b"")
        again = self.make()
        self.assertEqual(len(again.records), 5)

    def test_rotation_des_sauvegardes(self):
        s = self.make()
        s.add("premier souvenir")
        for _ in range(12):
            self.clock.advance(0.01)
            s.compact()
        backups = list((s.ns_dir / "backups").glob("*.jsonl.gz"))
        self.assertEqual(len(backups), store_mod.KEEP_BACKUPS)
        with gzip.open(backups[0], "rb") as f:
            self.assertIn(b"premier souvenir", f.read())

    def test_derniere_ligne_tronquee(self):
        s = self.make()
        _, rec = s.add("souvenir intact")
        s.close()
        with open(s.ns_dir / "events.jsonl", "ab") as f:
            f.write(b'{"op":"add","ts":"2026-01-01T00:00:00+00:00","r":{"id":"m_cass')    # écriture interrompue
        again = self.make()
        self.assertEqual(set(again.records), {rec["id"]})
        again.add("souvenir suivant")
        lines = (again.ns_dir / "events.jsonl").read_text("utf-8").splitlines()
        self.assertTrue(all(json.loads(l) for l in lines))        # plus aucune ligne cassée
        self.assertEqual(len(again.records), 2)

    def test_ligne_corrompue_au_milieu_ignoree(self):
        s = self.make()
        s.add("premier")
        s.close()
        with open(s.ns_dir / "events.jsonl", "ab") as f:
            f.write(b"ceci n'est pas du json\n")
        s2 = self.make()
        s2.add("deuxieme souvenir")
        s2.close()
        self.assertEqual(len(self.make().records), 2)

    def test_crash_entre_remplacement_et_troncature(self):
        s = self.make()
        s.add("souvenir a")
        s.add("souvenir b")
        # On simule : l'instantané est écrit mais le journal n'a pas été vidé
        snapshot_lines = [json.dumps(s._record_to_json(r), ensure_ascii=False) for r in s.records.values()]
        (s.ns_dir / "memories.jsonl").write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")
        s.close()
        again = self.make()
        self.assertEqual(len(again.records), 2)          # les événements rejoués ne créent pas de doublon

    def test_edition_manuelle_rechargee(self):
        s = self.make()
        s.add("souvenir original")
        rec = {"id": "m_aaaaaa", "t": "fait", "txt": "ajouté à la main", "imp": 5, "tags": [], "h": "",
               "c": store_mod.iso(self.clock()), "last": store_mod.iso(self.clock()), "hits": 0, "s": 1.0, "src": "main"}
        with open(s.ns_dir / "events.jsonl", "ab") as f:
            f.write((json.dumps({"op": "add", "ts": store_mod.iso(self.clock()), "r": rec}) + "\n").encode("utf-8"))
        res, _ = s.search("ajouté main")
        self.assertEqual(res[0]["id"], "m_aaaaaa")

    def test_limite_d_ecritures_du_llm(self):
        s = self.make()
        statuses = [s.add(f"fait distinct numero {i} zorglub", src="llm")[0] for i in range(store_mod.WRITES_PER_MINUTE + 3)]
        self.assertEqual(statuses.count("rate_limit"), 3)
        self.assertEqual(s.add("ajout par la page", src="ui")[0], "added")      # la page n'est pas limitée
        self.clock.advance(1 / 24 / 60 * 2)
        self.assertEqual(s.add("encore un fait", src="llm")[0], "added")

    def test_lecture_seule_refuse_les_ecritures(self):
        s = self.make()
        s.read_only = True
        self.assertEqual(s.add("x y z")[0], "locked")
        self.assertFalse(s.delete("m_000000"))

    def test_texte_borne(self):
        s = self.make()
        _, rec = s.add("a" * 2000)
        self.assertEqual(len(rec["txt"]), store_mod.MAX_TEXT)


class ActivityTests(StoreCase):
    def test_une_ligne_json_par_evenement(self):
        act = Activity(self.dirs / "logs")
        s = self.make(activity=act)
        s.add("Marion préfère 19 degrés", "préférence")
        s.search("Marion")
        events = [json.loads(l) for l in (self.dirs / "logs" / "activity.jsonl").read_text("utf-8").splitlines()]
        self.assertEqual([e["ev"] for e in events], ["start", "add", "search"])
        self.assertEqual(events[2]["n_res"], 1)
        self.assertNotIn("Marion préfère", json.dumps(events[1]))      # le texte du souvenir n'est pas recopié

    def test_empreinte_si_texte_desactive(self):
        act = Activity(self.dirs / "logs", log_text=False)
        s = self.make(activity=act)
        s.search("question secrete")
        log = (self.dirs / "logs" / "activity.jsonl").read_text("utf-8")
        self.assertNotIn("secrete", log)
        self.assertIn("sha:", log)

    def test_rotation(self):
        act = Activity(self.dirs / "logs", max_bytes=2000, keep=3)
        for i in range(200):
            act.log("x", i=i, pad="y" * 30)
        files = sorted(p.name for p in (self.dirs / "logs").iterdir())
        self.assertEqual(files, ["activity.1.jsonl", "activity.2.jsonl", "activity.jsonl"])
        for p in (self.dirs / "logs").iterdir():
            self.assertLessEqual(p.stat().st_size, 2000)


class BenchTests(StoreCase):
    @unittest.skipUnless(os.environ.get("MEMORY_BENCH"), "MEMORY_BENCH=1 pour lancer le benchmark")
    def test_10000_souvenirs(self):
        words = "maison chambre salon cuisine bureau chauffage volet lumiere porte garage jardin enfant famille".split()
        s = self.make()
        t0 = time.perf_counter()
        for i in range(10_000):
            s.add(f"souvenir {i} {words[i % 13]} {words[(i * 7) % 13]} reference {i * 31}", src="ui")
        add_ms = (time.perf_counter() - t0) * 1000 / 10_000
        s.compact()
        s.close()
        t0 = time.perf_counter()
        again = self.make()
        start_ms = (time.perf_counter() - t0) * 1000
        times = []
        for q in ["chauffage chambre", "volet salon 120", "famille jardin", "reference 4000", "porte garage"] * 20:
            t = time.perf_counter()
            again.search(q, touch=False)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        p95 = times[int(len(times) * 0.95)]
        print(f"\n[bench] ajout moyen {add_ms:.2f} ms | démarrage {start_ms:.0f} ms | recherche p95 {p95:.2f} ms")
        self.assertLess(start_ms, 1000)         # objectif 300-500 ms ; marge pour une machine chargée (le démarrage est unique)
        self.assertLess(p95, 10)
        self.assertLess(add_ms, 20)


if __name__ == "__main__":
    unittest.main()
