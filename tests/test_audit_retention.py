"""Rétention et purge du journal d'audit (components/audit_retention.py) : par âge (défaut et par accès), ciblée, journal des purges sans contenu."""
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import audit_retention as ar  # noqa: E402
from components import echanges  # noqa: E402

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc).timestamp()
JOUR = 86400


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="retention_"))
        echanges.configure(self.tmp / "audit", enabled=True, log_text=True, retention_days=90)
        echanges.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ecrire(self, age_jours, src="aria", who="Olivier", conv="c1", q="Une question secrète"):
        """Écrit un échange commencé il y a `age_jours` jours (à midi) dans le fichier de ce jour-là."""
        e = echanges.Echange(src, who, conv, query=q)
        e.ts = NOW - age_jours * JOUR
        echanges.write(e, "Réponse", "ok")
        return e

    def ids(self):
        return sorted(l["id"] for _, p in echanges.files() for l in echanges.read_lines(p))

    def jours(self):
        return [d for d, _ in echanges.files()]


class AgeTests(Base):
    def test_les_jours_trop_vieux_sont_supprimes_sans_toucher_aux_autres(self):
        vieux, recent = self.ecrire(120), self.ecrire(10)
        r = ar.purge_auto(90, now=NOW)
        self.assertEqual((r["lignes"], r["fichiers"]), (1, 1))
        self.assertEqual(self.ids(), [recent.id])
        self.assertNotIn(vieux.id, self.ids())

    def test_limite_exacte_jour_conserve_jusqu_a_la_fin_de_sa_periode(self):
        a = self.ecrire(89)
        self.ecrire(95)
        ar.purge_auto(90, now=NOW)
        self.assertEqual(self.ids(), [a.id])

    def test_idempotent(self):
        self.ecrire(200)
        ar.purge_auto(90, now=NOW)
        self.assertEqual(ar.purge_auto(90, now=NOW)["lignes"], 0)

    def test_duree_par_defaut_reglable(self):
        a, b = self.ecrire(40), self.ecrire(10)
        ar.purge_auto(30, now=NOW)
        self.assertEqual(self.ids(), [b.id])

    def test_un_acces_peut_exiger_moins_que_le_defaut(self):
        tiers_court = self.ecrire(20, src="k_aaaaaa:Robot", who="Robot")
        tiers_normal = self.ecrire(20, src="k_bbbbbb:Autre", who="Autre")
        aria = self.ecrire(20)
        jours = {"k_aaaaaa": 7}
        r = ar.purge_auto(90, access_days=lambda a: jours.get(a, 0), now=NOW)
        self.assertEqual(r["lignes"], 1)
        self.assertEqual(self.ids(), sorted([tiers_normal.id, aria.id]))
        self.assertNotIn(tiers_court.id, self.ids())

    def test_la_retention_par_acces_suit_l_identifiant_pas_le_nom(self):
        ancien_nom = self.ecrire(20, src="k_aaaaaa:Ancien nom", who="Robot")
        nouveau_nom = self.ecrire(20, src="k_aaaaaa:Nouveau nom", who="Robot")
        ar.purge_auto(90, access_days=lambda a: 7 if a == "k_aaaaaa" else 0, now=NOW)
        self.assertEqual(self.ids(), [])
        self.assertTrue(ancien_nom.id and nouveau_nom.id)

    def test_un_acces_ne_peut_pas_faire_conserver_plus_que_le_defaut_du_journal(self):
        """La durée du journal est un plafond : la rétention par accès ne fait que raccourcir."""
        self.ecrire(100, src="k_aaaaaa:Robot", who="Robot")
        ar.purge_auto(90, access_days=lambda a: 365, now=NOW)
        self.assertEqual(self.ids(), [])

    def test_aria_n_est_pas_touche_par_une_retention_d_acces(self):
        a = self.ecrire(20)
        ar.purge_auto(90, access_days=lambda acces: 1, now=NOW)
        self.assertEqual(self.ids(), [a.id])

    def test_fichier_reecrit_garde_les_autres_lignes_et_la_ligne_partielle(self):
        garde = self.ecrire(20)
        self.ecrire(20, src="k_aaaaaa:Robot", who="Robot")
        chemin = echanges.files()[0][1]
        with open(chemin, "ab") as f:
            f.write(b'{"id":"e_partiel","q":"coup')
        ar.purge_auto(90, access_days=lambda a: 1 if a == "k_aaaaaa" else 0, now=NOW)
        self.assertEqual(self.ids(), [garde.id])
        self.assertTrue(chemin.read_bytes().endswith(b'{"id":"e_partiel","q":"coup'))
        self.assertFalse(list(chemin.parent.glob("*.tmp")))

    def test_un_fichier_vide_apres_purge_est_supprime(self):
        self.ecrire(20, src="k_aaaaaa:Robot", who="Robot")
        ar.purge_auto(90, access_days=lambda a: 1, now=NOW)
        self.assertEqual(self.jours(), [])

    def test_une_erreur_dans_la_lecture_des_reglages_utilise_le_defaut(self):
        self.ecrire(100)
        keep = self.ecrire(5, src="k_aaaaaa:R", who="R")

        def casse(a):
            raise TypeError("x")
        ar.purge_auto(90, access_days=lambda a: "abc", now=NOW)
        self.assertEqual(self.ids(), [keep.id])

    def test_ecriture_pendant_la_purge_n_est_jamais_perdue(self):
        for _ in range(20):
            self.ecrire(100)
        nouveaux = []

        def ecrire():
            for _ in range(40):
                nouveaux.append(self.ecrire(0).id)
        t = threading.Thread(target=ecrire)
        t.start()
        ar.purge_auto(90, now=NOW)
        t.join()
        self.assertEqual(sorted(nouveaux), self.ids())


class CibleeTests(Base):
    def test_au_moins_un_critere(self):
        self.ecrire(1)
        with self.assertRaises(ValueError):
            ar.purge_targeted()
        self.assertEqual(len(self.ids()), 1)

    def test_par_conversation(self):
        a = self.ecrire(1, conv="c1")
        b = self.ecrire(1, conv="c2")
        r = ar.purge_targeted(conversation="c1")
        self.assertEqual(r["lignes"], 1)
        self.assertEqual(self.ids(), [b.id])
        self.assertNotIn(a.id, self.ids())

    def test_par_acces_identifiant(self):
        self.ecrire(1, src="k_aaaaaa:Robot", who="Robot")
        garde = [self.ecrire(1, src="k_bbbbbb:Autre", who="Autre"), self.ecrire(1)]
        ar.purge_targeted(acces="k_aaaaaa")
        self.assertEqual(self.ids(), sorted(e.id for e in garde))

    def test_par_acces_aria(self):
        self.ecrire(1)
        tiers = self.ecrire(1, src="k_aaaaaa:Robot", who="Robot")
        ar.purge_targeted(acces="aria")
        self.assertEqual(self.ids(), [tiers.id])

    def test_par_personne_insensible_a_la_casse_et_exacte(self):
        self.ecrire(1, who="Olivier")
        autre = self.ecrire(1, who="Olivier Martin")
        ar.purge_targeted(personne="  olivier ")
        self.assertEqual(self.ids(), [autre.id])

    def test_par_date_exclut_le_jour_donne(self):
        self.ecrire(30)
        recent = self.ecrire(2)
        ar.purge_targeted(date_avant="2026-11-12")           # le 12 (jour de `recent`) est exclu
        self.assertEqual(self.ids(), [recent.id])

    def test_date_invalide(self):
        for faux in ("2026-13-01", "hier", "2026/11/01", "'; drop"):
            with self.assertRaises(ValueError):
                ar.purge_targeted(date_avant=faux)

    def test_criteres_combines_en_et(self):
        a = self.ecrire(1, conv="c1", who="Olivier")
        b = self.ecrire(1, conv="c1", who="Paul")
        ar.purge_targeted(conversation="c1", personne="Paul")
        self.assertEqual(self.ids(), [a.id])
        self.assertNotIn(b.id, self.ids())

    def test_retourne_ids_et_periode_sans_contenu(self):
        e = self.ecrire(3)
        r = ar.purge_targeted(conversation="c1")
        self.assertEqual(r["ids"], [e.id])
        self.assertTrue(r["periode"][0].startswith("2026-11-11"))


class JournalDesPurgesTests(Base):
    def test_une_ligne_par_purge_sans_aucun_contenu(self):
        self.ecrire(3, q="Mon secret : 0612345678", who="Léa Dupont", conv="conv-secrete")
        ar.purge_targeted(conversation="conv-secrete", personne="Léa Dupont")
        (p,) = ar.read_purges()
        brut = json.dumps(p, ensure_ascii=False)
        for secret in ("0612345678", "Léa", "Dupont", "Réponse"):
            self.assertNotIn(secret, brut)
        self.assertEqual((p["par"], p["lignes"]), ("admin", 1))
        self.assertIn("conversation", p["critere"])
        for champ in ("ts", "critere", "lignes", "fichiers", "du", "au"):
            self.assertIn(champ, p)

    def test_la_purge_automatique_est_journalisee_et_rien_si_rien_a_purger(self):
        ar.purge_auto(90, now=NOW)
        self.assertEqual(ar.read_purges(), [])
        self.ecrire(200)
        ar.purge_auto(90, now=NOW)
        (p,) = ar.read_purges()
        self.assertEqual((p["par"], p["lignes"], p["fichiers"]), ("auto", 1, 1))

    def test_le_journal_des_purges_n_est_pas_pris_pour_un_journal_d_echanges(self):
        self.ecrire(200)
        ar.purge_auto(90, now=NOW)
        self.assertEqual([n for _, n in echanges.files() if n.name == ar.PURGES_FILE], [])

    def test_une_purge_a_zero_ligne_est_aussi_tracee(self):
        self.ecrire(1)
        ar.purge_targeted(conversation="inexistante")
        self.assertEqual(ar.read_purges()[-1]["lignes"], 0)

    def test_les_purges_ne_sont_pas_purgees_par_la_retention(self):
        self.ecrire(200)
        ar.purge_auto(90, now=NOW)
        ar.purge_auto(1, now=NOW + 400 * JOUR)
        self.assertEqual(len(ar.read_purges()), 1)


class VolumesTests(Base):
    def test_volumes_par_jour_sans_contenu(self):
        self.ecrire(1)
        self.ecrire(1)
        self.ecrire(5)
        v = ar.volumes()
        self.assertEqual(sorted(x["lignes"] for x in v), [1, 2])
        self.assertNotIn("Une question secrète", json.dumps(v))


class PlanificateurTests(Base):
    def test_une_passe_utilise_les_reglages_du_moment(self):
        self.ecrire(40)
        reglages = {"jours": 90}
        p = ar.Planificateur(lambda: (reglages["jours"], None))
        self.assertEqual(p.passe()["lignes"], 0)
        reglages["jours"] = 30
        self.assertEqual(p.passe()["lignes"], 1)

    def test_une_erreur_ne_tue_pas_le_planificateur(self):
        def casse():
            raise RuntimeError("boum")
        p = ar.Planificateur(casse)
        self.assertIsNone(p.passe())

    def test_demarre_en_thread_demon_et_purge_aussitot(self):
        e = self.ecrire(0)
        a_purger = echanges.Echange("aria", "x", "c")
        a_purger.ts = time.time() - 200 * JOUR
        echanges.write(a_purger, "r")
        p = ar.Planificateur(lambda: (90, None), interval=3600).start()
        for _ in range(100):
            if a_purger.id not in self.ids():
                break
            time.sleep(0.05)
        p.stop()
        self.assertTrue(p.thread.daemon)
        self.assertIn(e.id, self.ids())
        self.assertNotIn(a_purger.id, self.ids())


if __name__ == "__main__":
    unittest.main()
