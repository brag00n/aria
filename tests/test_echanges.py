"""Journal d'audit des échanges (components/echanges.py) et son point d'écriture (Llm.get_answer_web). Bibliothèque standard seulement
(un faux moteur remplace le LLM)."""
import asyncio
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import echanges  # noqa: E402


class FauxMoteur:
    def __init__(self, morceaux, erreur=None, attente=0.0):
        self.morceaux, self.erreur, self.attente = morceaux, erreur, attente

    async def get_answer_web(self, tts, query, user, contexte=""):
        for i, m in enumerate(self.morceaux):
            if self.attente:
                await asyncio.sleep(self.attente)
            yield m, None, None
            if self.erreur and i == 0:
                raise self.erreur


def faux_llm(moteur):
    """Un `Llm` sans moteur réel : seule la méthode du journal est testée."""
    try:
        from components.llm import Llm
    except ImportError:
        return None
    obj = Llm.__new__(Llm)
    obj.engine, obj.backend_type = moteur, "anthropic"
    return obj


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="echanges_"))
        echanges.configure(self.tmp / "audit", enabled=True, log_text=True, retention_days=90)
        echanges.clear()

    def tearDown(self):
        echanges.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lignes(self):
        out = []
        for _, p in echanges.files():
            out.extend(echanges.read_lines(p))
        return out


class EcritureTests(Base):
    def ecrire(self, **kw):
        e = echanges.start(kw.pop("src", "aria"), kw.pop("who", "Olivier"), kw.pop("conv", "c1"), voix=kw.pop("voix", False), query=kw.pop("query", "Bonjour"))
        echanges.write(e, kw.pop("reponse", "Salut"), kw.pop("statut", "ok"))
        return e

    def test_une_ligne_par_echange_avec_tous_les_champs(self):
        e = self.ecrire(src="k_2b7cfa:Robot", who="Robot", conv="k_2b7cfa:bamboo#1", voix=True, query="Quelle heure est-il ?", reponse="Il est midi.")
        (l,) = self.lignes()
        self.assertEqual((l["id"], l["src"], l["who"], l["conv"], l["q"], l["r"], l["st"], l["voix"]),
                         (e.id, "k_2b7cfa:Robot", "Robot", "k_2b7cfa:bamboo#1", "Quelle heure est-il ?", "Il est midi.", "ok", True))
        for champ in ("ts", "fin", "ms", "outils"):
            self.assertIn(champ, l)
        self.assertTrue(echanges.valid_id(l["id"]))

    def test_un_fichier_par_jour_nomme_par_la_date_du_debut(self):
        self.ecrire()
        fichiers = echanges.files()
        self.assertEqual(len(fichiers), 1)
        self.assertRegex(fichiers[0][1].name, r"^echanges-\d{4}-\d{2}-\d{2}\.jsonl$")
        self.assertEqual(fichiers[0][0], self.lignes()[0]["ts"][:10])

    def test_les_lignes_s_ajoutent_sans_ecraser(self):
        for i in range(5):
            self.ecrire(query=f"Question {i}")
        self.assertEqual([l["q"] for l in self.lignes()], [f"Question {i}" for i in range(5)])

    def test_accents_et_guillemets(self):
        self.ecrire(query='Il a dit « Léa, c\'est ça » à Zoé', reponse="Très bien — noté.")
        (l,) = self.lignes()
        self.assertEqual((l["q"], l["r"]), ('Il a dit « Léa, c\'est ça » à Zoé', "Très bien — noté."))

    def test_les_marqueurs_d_outil_sont_retires_de_la_reponse(self):
        self.ecrire(reponse="Je cherche.\n\n> 🛠️ Outil utilisé : `meteo`\n\nIl pleut.")
        self.assertNotIn("🛠️", self.lignes()[0]["r"])
        self.assertIn("Il pleut.", self.lignes()[0]["r"])

    def test_aucune_ecriture_si_desactive(self):
        echanges.configure(self.tmp / "audit", enabled=False)
        self.ecrire()
        self.assertEqual(self.lignes(), [])

    def test_option_empreinte_seulement(self):
        echanges.configure(self.tmp / "audit", enabled=True, log_text=False)
        self.ecrire(query="Mon numéro est 0612345678", reponse="Noté.")
        (l,) = self.lignes()
        self.assertNotIn("0612345678", json.dumps(l))
        self.assertTrue(l["q"].startswith("sha:") and l["r"].startswith("sha:"))
        self.assertEqual(l["src"], "aria")                                           # les métadonnées restent

    def test_une_erreur_d_ecriture_ne_leve_jamais(self):
        echanges.configure(self.tmp / "fichier_pas_dossier")
        (self.tmp / "fichier_pas_dossier").write_text("x", "utf-8")                 # le « dossier » est un fichier : l'écriture échoue
        e = echanges.start("aria", "x", "c")
        self.assertFalse(echanges.write(e, "r"))
        self.assertFalse(echanges.write(None, "r"))

    def test_ligne_partielle_ignoree_a_la_lecture(self):
        self.ecrire()
        p = echanges.files()[0][1]
        with open(p, "ab") as f:
            f.write(b'{"id":"e_partiel","q":"coup')
        self.assertEqual(len(echanges.read_lines(p)), 1)

    def test_lignes_illisibles_ignorees(self):
        self.ecrire()
        p = echanges.files()[0][1]
        with open(p, "ab") as f:
            f.write(b"pas du json\n[1]\n")
        self.assertEqual(len(echanges.read_lines(p)), 1)

    def test_ecritures_concurrentes_sans_ligne_melangee(self):
        def ecrire(n):
            for k in range(30):
                self.ecrire(query=f"Question {n}-{k}" + "x" * 500)
        ts = [threading.Thread(target=ecrire, args=(n,)) for n in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        brut = echanges.files()[0][1].read_bytes().split(b"\n")[:-1]
        self.assertEqual(len(brut), 180)
        for ligne in brut:
            json.loads(ligne)                                                         # toutes les lignes sont du JSON complet
        self.assertEqual(len({l["id"] for l in self.lignes()}), 180)

    def test_unicite_garantie_meme_avec_un_alea_identique(self):
        """Régression : 24 bits d'aléa seuls pouvaient collisionner dans la même milliseconde (échec rare observé)."""
        import unittest.mock as mock
        with mock.patch.object(echanges.secrets, "token_hex", return_value="abcd"), mock.patch.object(echanges.time, "time", return_value=1_800_000_000.123):
            ids = [echanges.new_id() for _ in range(300)]
        self.assertEqual(len(set(ids)), 300)
        self.assertTrue(all(echanges.valid_id(i) for i in ids))

    def test_unicite_entre_threads(self):
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(16) as pool:
            ids = [i for lot in pool.map(lambda _: [echanges.new_id() for _ in range(500)], range(16)) for i in lot]
        self.assertEqual(len(set(ids)), len(ids))

    def test_identifiants_uniques_et_valides(self):
        ids = {echanges.new_id() for _ in range(2000)}
        self.assertEqual(len(ids), 2000)
        self.assertTrue(all(echanges.valid_id(i) for i in ids))
        for faux in ("", None, "e_", "x_123456", "e_" + "z" * 50, "e_12 34", 42, "e_../../x"):
            self.assertFalse(echanges.valid_id(faux), faux)


class SourceTests(unittest.TestCase):
    def test_source_aria_ou_identifiant_et_nom(self):
        self.assertEqual(echanges.source_label(None), "aria")
        self.assertEqual(echanges.source_label(""), "aria")
        self.assertEqual(echanges.source_label("k_2b7cfa", "Robot"), "k_2b7cfa:Robot")
        self.assertEqual(echanges.source_label("k_2b7cfa", "  Robot   du  garage "), "k_2b7cfa:Robot du garage")
        self.assertEqual(echanges.source_label("k_2b7cfa", ""), "k_2b7cfa")

    def test_identifiant_d_acces_lu_dans_la_source(self):
        self.assertEqual(echanges.acces_id_of("k_2b7cfa:Robot"), "k_2b7cfa")
        self.assertEqual(echanges.acces_id_of("k_2b7cfa"), "k_2b7cfa")
        self.assertEqual(echanges.acces_id_of("aria"), "")
        self.assertEqual(echanges.acces_id_of(""), "")
        self.assertEqual(echanges.acces_id_of(None), "")

    def test_le_nom_peut_contenir_deux_points_sans_perdre_l_identifiant(self):
        self.assertEqual(echanges.acces_id_of(echanges.source_label("k_1", "Salle: A")), "k_1")

    def test_extrait(self):
        self.assertEqual(echanges.extrait("  a \n b  "), "a b")
        long = "mot " * 100
        self.assertLessEqual(len(echanges.extrait(long)), echanges.EXTRAIT_MAX)
        self.assertTrue(echanges.extrait(long).endswith("…"))
        self.assertEqual(echanges.extrait(None), "")

    def test_meta_porte_ce_qu_il_faut_pour_la_trace_sans_donnee_du_llm(self):
        echanges.clear()
        self.assertIsNone(echanges.current_meta())
        echanges.start("k_1:Robot", "Robot", "c", query="Retiens que la batterie est à 80 %")
        m = echanges.current_meta()
        self.assertEqual(set(m), {"echange", "ts", "src", "who", "x"})
        self.assertTrue(echanges.valid_id(m["echange"]))
        self.assertEqual((m["src"], m["who"], m["x"]), ("k_1:Robot", "Robot", "Retiens que la batterie est à 80 %"))
        echanges.clear()


class ConversationAriaTests(unittest.TestCase):
    def test_un_nouvel_identifiant_a_chaque_renouvellement(self):
        c = echanges.ConversationAria()
        a = c.id
        b = c.renew()
        self.assertNotEqual(a, b)
        self.assertEqual(c.id, b)
        self.assertTrue(c.id.startswith("aria-"))
        self.assertNotEqual(echanges.ConversationAria().renew(), c.renew())


class PointDEcritureTests(Base):
    """Llm.get_answer_web écrit UNE ligne par échange, pour tous les moteurs, y compris sur arrêt et erreur."""

    def lancer(self, moteur, requete="Bonjour", interrompre=False, **meta):
        llm = faux_llm(moteur)
        if llm is None:
            self.skipTest("anthropic/openai absents de l'hôte : testé dans le conteneur")
        echanges.start(meta.get("src", "aria"), meta.get("who", "Olivier"), meta.get("conv", "c1"), voix=meta.get("voix", False))

        async def tour():
            gen = llm.get_answer_web(None, requete, "u")
            async for _ in gen:
                if interrompre:
                    await gen.aclose()
                    return
        try:
            asyncio.run(tour())
        except Exception:
            pass

    def test_une_ligne_a_la_fin_du_tour_avec_question_et_reponse(self):
        self.lancer(FauxMoteur(["Bon", "Bonjour à vous"]), requete="Salut Aria")
        (l,) = self.lignes()
        self.assertEqual((l["q"], l["r"], l["st"], l["src"], l["who"]), ("Salut Aria", "Bonjour à vous", "ok", "aria", "Olivier"))

    def test_arret_en_cours_de_reponse(self):
        self.lancer(FauxMoteur(["Début", "Début de réponse"]), interrompre=True)
        (l,) = self.lignes()
        self.assertEqual((l["st"], l["r"]), ("interrompu", "Début"))                 # la réponse partielle est conservée

    def test_erreur_du_moteur(self):
        self.lancer(FauxMoteur(["Partiel", "Partiel suite"], erreur=RuntimeError("boum")))
        (l,) = self.lignes()
        self.assertEqual((l["st"], l["r"]), ("erreur", "Partiel"))

    def test_erreur_signalee_par_le_texte_du_moteur(self):
        self.lancer(FauxMoteur(["Erreur LLM: quota dépassé"]))
        self.assertEqual(self.lignes()[0]["st"], "erreur")

    def test_reponse_vide_du_modele_est_une_erreur_et_pas_un_silence(self):
        for morceaux in ([], [""], ["   "], ["\n\n> 🛠️ Outil utilisé : `diagnostics`\n\n"]):          # aucun texte, ou seulement la ligne « outil utilisé »
            with self.subTest(morceaux=morceaux):
                for f in list(self.tmp.rglob("*.jsonl")):
                    f.unlink()
                self.lancer(FauxMoteur(morceaux))
                (l,) = self.lignes()
                self.assertEqual(l["st"], "erreur")
                self.assertTrue(l["r"].startswith("Erreur LLM:"))
                self.assertEqual(l["r"], "Erreur LLM: (pas de réponse)")                   # aucun détail technique dans ce que reçoit le client ni dans l'audit

    def test_la_reponse_vide_est_signalee_dans_les_logs_avec_le_modele(self):
        import components.utils as utils
        vus = []
        original = utils.log_info
        utils.log_info = lambda tag, msg, *a, **k: vus.append((tag, msg))
        try:
            m = FauxMoteur([])
            m.model_name = "petit-modele"
            self.lancer(m)
        finally:
            utils.log_info = original
        avert = [msg for tag, msg in vus if "Réponse vide" in msg]
        self.assertEqual(len(avert), 1)
        self.assertIn("petit-modele", avert[0])
        self.assertIn("message système", avert[0])

    def test_la_reponse_vide_est_aussi_transmise_a_l_appelant_pas_seulement_au_journal(self):
        llm = faux_llm(FauxMoteur([]))
        if llm is None:
            self.skipTest("anthropic/openai absents de l'hôte")
        echanges.start("aria", "Olivier", "c1")
        vus = []

        async def tour():
            async for values in llm.get_answer_web(None, "q", "u"):
                vus.append(values[0])
        asyncio.run(tour())
        self.assertEqual(len(vus), 1)
        self.assertTrue(vus[0].startswith("Erreur LLM:"))                               # le tour d'agent la transforme en événement `error` (turn.py)

    def test_une_erreur_du_moteur_apres_du_texte_n_ajoute_pas_de_reponse_vide(self):
        self.lancer(FauxMoteur(["Partiel", "Partiel suite"], erreur=RuntimeError("boum")))
        (l,) = self.lignes()
        self.assertEqual((l["st"], l["r"]), ("erreur", "Partiel"))                      # la réponse partielle reste telle quelle

    def test_une_reponse_normale_ou_courte_n_est_pas_une_erreur(self):
        for texte in ("Ok", "?", "Oui."):
            with self.subTest(texte=texte):
                for f in list(self.tmp.rglob("*.jsonl")):
                    f.unlink()
                self.lancer(FauxMoteur([texte]))
                self.assertEqual(self.lignes()[0]["st"], "ok")

    def test_un_arret_sans_texte_reste_interrompu_pas_une_erreur_de_reponse_vide(self):
        self.lancer(FauxMoteur(["", "x"]), interrompre=True)
        self.assertEqual(self.lignes()[0]["st"], "interrompu")

    def test_hors_echange_aucune_ecriture(self):
        llm = faux_llm(FauxMoteur(["x"]))
        if llm is None:
            self.skipTest("anthropic/openai absents de l'hôte")
        echanges.clear()

        async def tour():
            async for _ in llm.get_answer_web(None, "q", "u"):
                pass
        asyncio.run(tour())
        self.assertEqual(self.lignes(), [])

    def test_tours_concurrents_chacun_sa_ligne_et_ses_metadonnees(self):
        llm = faux_llm(FauxMoteur(["a", "ab"], attente=0.01))
        if llm is None:
            self.skipTest("anthropic/openai absents de l'hôte")

        async def tour(n):
            echanges.start(f"k_{n}:Acces{n}", f"Acces{n}", f"conv{n}", query=f"question {n}")
            async for _ in llm.get_answer_web(None, f"question {n}", "u"):
                await asyncio.sleep(0.005 * (6 - n))                                  # les tours se croisent
            return echanges.current().id

        async def tous():
            return await asyncio.gather(*(tour(n) for n in range(1, 6)))
        ids = asyncio.run(tous())
        par_id = {l["id"]: l for l in self.lignes()}
        self.assertEqual(len(par_id), 5)
        for n, eid in zip(range(1, 6), ids):
            l = par_id[eid]
            self.assertEqual((l["src"], l["who"], l["conv"], l["q"]), (f"k_{n}:Acces{n}", f"Acces{n}", f"conv{n}", f"question {n}"))

    def test_la_question_est_completee_si_la_source_ne_l_a_pas_donnee(self):
        self.lancer(FauxMoteur(["ok"]), requete="Question du moteur")
        self.assertEqual(self.lignes()[0]["q"], "Question du moteur")


if __name__ == "__main__":
    unittest.main()
