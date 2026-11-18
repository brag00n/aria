"""Cœur d'un tour (components/agents/turn.py) : événements bruts, sans SSE ni HTTP. Les formateurs (SSE de l'API, chat de l'interface) le consomment tels quels.
Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from components.agents import sessions, turn
    from components import echanges
    HAVE = True
except ImportError:
    HAVE = False


class Moteur:
    def __init__(self, morceaux, wav=None, erreur=None, lent=0.0):
        self.morceaux, self.wav, self.erreur, self.lent = morceaux, wav, erreur, lent
        self.appels = []

    async def get_answer_web(self, tts, query, user, contexte=""):
        self.appels.append({"query": query, "user": user, "contexte": contexte, "tts": tts is not None})
        for i, m in enumerate(self.morceaux):
            if self.lent:
                await asyncio.sleep(self.lent)
            if self.erreur and i == 1:
                raise self.erreur
            yield m, (self.wav if i == len(self.morceaux) - 1 else None), "dit"


class Session:
    def __init__(self, moteur, conv=None, speaker=None):
        self.engine, self.conv, self.speaker = moteur, conv, speaker
        self.sid, self.user, self.epoch = "k_1:s", "k_1:s", 1
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()

    def touch(self):
        pass

    @property
    def conv_id(self):
        return f"{self.sid}#{self.epoch}"


class Deps:
    def __init__(self, texte_stt="quelle heure", tts=True):
        self.tts, self.stt, self.stt_lock, self.max_audio_s = (object() if tts else None), self, None, 30.0
        self.texte_stt = texte_stt
        self.sessions = self
        self.resets = 0

    def decode_audio(self, raw):
        return [0.0] * 16000

    def transcribe_translate(self, samples):
        return self.texte_stt

    def reset(self, session):
        self.resets += 1
        session.epoch += 1


REC = {"id": "k_1", "label": "Robot", "publish_speaker": False}


async def collecter(deps, session, texte="bonjour", audio=None, speak=True, rec=None, **kw):
    return [e async for e in turn.run_turn(deps, session, rec or REC, None if audio else texte, audio, speak, lambda *a, **k: None, **kw)]


def lancer(coro):
    return asyncio.run(coro)


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class CoeurTests(unittest.TestCase):
    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        echanges.configure(None, enabled=False)
        echanges.clear()

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def noms(self, ev):
        return [n for n, _ in ev]

    def test_un_tour_texte_normal(self):
        m = Moteur(["Bon", "Bonjour"])
        ev = lancer(collecter(Deps(), Session(m)))
        self.assertEqual(self.noms(ev), ["state", "text", "text", "done"])
        self.assertEqual(ev[0][1], {"state": "thinking"})
        self.assertEqual([d["text"] for n, d in ev if n == "text"], ["Bon", "Bonjour"])
        self.assertIn("ms", ev[-1][1])

    def test_le_texte_n_est_emis_qu_en_cas_de_changement(self):
        ev = lancer(collecter(Deps(), Session(Moteur(["A", "A", "AB"]))))
        self.assertEqual([d["text"] for n, d in ev if n == "text"], ["A", "AB"])

    def test_un_tour_vocal_emet_user_text_et_passe_la_voix_a_l_audit(self):
        echanges.configure(None, enabled=False)
        deps = Deps("quelle heure est-il")
        ev = lancer(collecter(deps, Session(Moteur(["Il est midi"])), audio="data:audio/wav;base64,AAAA"))
        self.assertEqual(self.noms(ev)[:2], ["state", "user_text"])
        self.assertEqual(ev[1][1], {"text": "quelle heure est-il"})

    def test_le_chemin_du_wav_est_renvoye_tel_quel_sans_lecture(self):
        ev = lancer(collecter(Deps(), Session(Moteur(["Salut"], wav="/tmp/inexistant.wav"))))
        audio = [d for n, d in ev if n == "audio"]
        self.assertEqual(audio, [{"path": "/tmp/inexistant.wav", "text": "dit"}])
        self.assertIn("speaking", [d.get("state") for n, d in ev if n == "state"])

    def test_sans_synthese_le_moteur_recoit_tts_none(self):
        m = Moteur(["x"])
        lancer(collecter(Deps(), Session(m), speak=False))
        self.assertFalse(m.appels[0]["tts"])
        m2 = Moteur(["x"])
        lancer(collecter(Deps(), Session(m2), speak=True))
        self.assertTrue(m2.appels[0]["tts"])

    def test_le_contexte_est_transmis_seulement_s_il_existe(self):
        m = Moteur(["x"])
        lancer(collecter(Deps(), Session(m), contexte="Interlocuteur actuel : Léa."))
        self.assertEqual(m.appels[0]["contexte"], "Interlocuteur actuel : Léa.")
        m2 = Moteur(["x"])
        lancer(collecter(Deps(), Session(m2)))
        self.assertEqual(m2.appels[0]["contexte"], "")

    def test_un_moteur_sans_parametre_contexte_reste_appelable(self):
        """Les moteurs des clients actuels n'ont pas toujours le 4e paramètre : sans contexte, l'appel ne change pas."""
        class Ancien:
            async def get_answer_web(self, tts, query, user):
                yield "ok", None, None
        ev = lancer(collecter(Deps(), Session(Ancien())))
        self.assertEqual(self.noms(ev)[-1], "done")

    def test_les_lignes_outil_deviennent_des_evenements_tool_et_sont_retirees_du_texte(self):
        ligne = "\n\n> 🛠️ Outil utilisé : `meteo`\n\n"
        ev = lancer(collecter(Deps(), Session(Moteur(["Je cherche" + ligne, "Je cherche" + ligne + " Il pleut"]))))
        self.assertEqual([d for n, d in ev if n == "tool"], [{"name": "meteo"}])                  # une seule fois
        self.assertTrue(all("🛠️" not in d["text"] for n, d in ev if n == "text"))

    def test_erreur_du_modele(self):
        ev = lancer(collecter(Deps(), Session(Moteur(["Erreur LLM: quota dépassé"]))))
        self.assertEqual(ev[-1], ("error", {"code": "llm", "message": "le modèle n'a pas pu répondre"}))
        self.assertNotIn("done", self.noms(ev))

    def test_reponse_vide_du_modele_le_client_ne_voit_que_pas_de_reponse(self):
        import components.utils as u
        ev = lancer(collecter(Deps(), Session(Moteur([u.NO_ANSWER_ERROR]))))
        self.assertEqual(ev[-1], ("error", {"code": "llm", "message": "(pas de réponse)"}))
        texte = json.dumps(ev)
        for mot in ("modèle", "Erreur LLM", "passerelle", "finish_reason", "message système"):
            self.assertNotIn(mot, texte)                                                              # aucun détail technique (rétro-ingénierie, préparation d'une attaque)
        self.assertNotIn("done", self.noms(ev))

    def test_exception_dans_le_moteur(self):
        ev = lancer(collecter(Deps(), Session(Moteur(["a", "b"], erreur=RuntimeError("boum")))))
        self.assertEqual(ev[-1], ("error", {"code": "internal", "message": "erreur interne"}))
        self.assertNotIn("boum", json.dumps(ev))                                                    # jamais de détail interne au client

    def test_un_tour_se_termine_toujours_par_done_ou_error_jamais_les_deux(self):
        for m in (Moteur(["a"]), Moteur(["Erreur LLM: x"]), Moteur(["a", "b"], erreur=RuntimeError("x"))):
            noms = self.noms(lancer(collecter(Deps(), Session(m))))
            self.assertEqual(sum(n in ("done", "error") for n in noms), 1)
            self.assertIn(noms[-1], ("done", "error"))

    def test_arret_demande_pendant_la_generation(self):
        async def scenario():
            s = Session(Moteur(["a", "b", "c", "d"], lent=0.05))
            sortie = []
            async for e in turn.run_turn(Deps(), s, REC, "x", None, False, lambda *a, **k: None):
                sortie.append(e)
                if e[0] == "text":
                    s.stop.set()
            return sortie
        ev = lancer(scenario())
        self.assertEqual(ev[-1], ("done", {"stopped": True}))

    def test_un_deuxieme_tour_simultane_recoit_busy(self):
        async def scenario():
            s = Session(Moteur(["a", "b"], lent=0.1))
            premier = asyncio.ensure_future(collecter(Deps(), s))
            await asyncio.sleep(0.03)
            second = await collecter(Deps(), s)
            return second, await premier
        second, premier = lancer(scenario())
        self.assertEqual(second, [("error", {"code": "busy", "message": "un tour est déjà en cours dans cette session"})])
        self.assertEqual(self.noms(premier)[-1], "done")

    def test_le_verrou_est_libere_apres_chaque_issue(self):
        s = Session(Moteur(["a", "b"], erreur=RuntimeError("x")))
        lancer(collecter(Deps(), s))
        self.assertFalse(s.lock.locked())

    def test_audio_trop_long_et_audio_illisible(self):
        class Long(Deps):
            def decode_audio(self, raw):
                return [0.0] * (16000 * 31)

        class Casse(Deps):
            def decode_audio(self, raw):
                raise ValueError("format")
        ev = lancer(collecter(Long(), Session(Moteur(["x"])), audio="data:audio/wav;base64,AAAA"))
        self.assertEqual(ev[-1][1]["code"], "audio_too_long")
        ev = lancer(collecter(Casse(), Session(Moteur(["x"])), audio="data:audio/wav;base64,AAAA"))
        self.assertEqual(ev[-1][1]["code"], "bad_audio")
        ev = lancer(collecter(Deps(), Session(Moteur(["x"])), audio="pas une data uri"))
        self.assertEqual(ev[-1][1]["code"], "bad_audio")

    def test_audio_sans_parole_donne_done_empty_sans_appeler_le_moteur(self):
        m = Moteur(["x"])
        ev = lancer(collecter(Deps("  ...  "), Session(m), audio="data:audio/wav;base64,AAAA"))
        self.assertEqual(ev[-1], ("done", {"empty": True}))
        self.assertEqual(m.appels, [])

    def test_consignes_ignorees_avertissement(self):
        ev = lancer(collecter(Deps(), Session(Moteur(["x"])), ignored=True))
        self.assertEqual(ev[1][0], "warning")
        self.assertEqual(ev[1][1]["code"], "system_ignored")

    # ---- audit : qui parle
    def test_qui_est_par_defaut_le_nom_de_l_agent_ou_l_interlocuteur_de_la_session(self):
        for rec, session, attendu in ((REC, Session(Moteur(["x"])), "Robot"),
                                      ({**REC, "publish_speaker": True}, Session(Moteur(["x"]), speaker={"nom": "Léa Dupont"}), "Léa Dupont"),
                                      ({**REC, "publish_speaker": True}, Session(Moteur(["x"])), "Inconnue")):
            capture = []
            orig = echanges.start
            echanges.start = lambda src, who, conv, voix=False, query="": capture.append((src, who)) or orig(src, who, conv, voix, query)
            try:
                lancer(collecter(Deps(), session, rec=rec))
            finally:
                echanges.start = orig
            self.assertEqual(capture[0], ("k_1:Robot", attendu))

    def test_qui_fourni_par_l_appelant_l_emporte(self):
        capture = []
        orig = echanges.start
        echanges.start = lambda src, who, conv, voix=False, query="": capture.append(who) or orig(src, who, conv, voix, query)
        try:
            lancer(collecter(Deps(), Session(Moteur(["x"])), qui="Olivier"))
        finally:
            echanges.start = orig
        self.assertEqual(capture, ["Olivier"])

    # ---- conversation à mots-clés dans le cœur
    def test_mots_cles_dans_le_coeur(self):
        conv = sessions.make_conversation({"enabled": True, "open_phrases": ["bonjour aria"], "close_phrases": ["merci aria"], "stop_phrases": ["stop"]})
        deps, m = Deps(), Moteur(["ok"])
        s = Session(m, conv=conv)
        self.assertEqual(lancer(collecter(deps, s, texte="stop"))[-1], ("done", {"stopped": True, "reason": "keyword"}))
        self.assertTrue(s.stop.is_set())
        self.assertEqual(m.appels, [])
        deps.texte_stt = "une phrase au hasard"
        self.assertEqual(lancer(collecter(deps, s, audio="data:audio/wav;base64,AAAA"))[-1], ("done", {"ignored": True}))
        deps.texte_stt = "bonjour aria"
        ev = lancer(collecter(deps, s, audio="data:audio/wav;base64,AAAA"))
        self.assertIn(("conversation", {"state": "open"}), ev)
        deps.texte_stt = "merci aria"
        ev = lancer(collecter(deps, s, audio="data:audio/wav;base64,AAAA"))
        self.assertEqual(self.noms(ev)[-2:], ["conversation", "done"])
        self.assertEqual(deps.resets, 1)


if __name__ == "__main__":
    unittest.main()
