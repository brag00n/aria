"""Chat de l'interface parlant à l'agent `aria` en interne (components/agents/chat.py) : interlocuteur global, mots-clés, arrêt, oubli, audit, tour.
Sans Gradio, sans HTTP. Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import asyncio
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from components import echanges
    from components.agents import chat, keys, profile, server, sessions
    HAVE = True
except ImportError:
    HAVE = False

CONFIG = {
    "active_modules": {"llm": "Llm_Anthropic"},
    "llm": {"Llm_Anthropic": {"params": {"backend_type": "anthropic"}}},
    "mcp": {"params": {"servers": {"web": {"url": "http://x/mcp"}}}},
    "memory": {"reset_on_close": True},
    "conversation": {"enabled": True, "open_phrases": ["bonjour aria"], "close_phrases": ["merci aria"], "stop_phrases": ["stop"]},
    "agents": {"profiles": {}},
}


class Moteur:
    """Moteur de fond (ce que `Llm.engine` est dans Aria) : garde l'historique par utilisateur."""

    def __init__(self, parent):
        self.parent = parent
        self.user_aware_messages = {}

    async def get_answer_web(self, tts, query, user, contexte=""):
        p = self.parent
        p.appels.append({"query": query, "user": user, "contexte": contexte, "tts": tts is not None})
        self.user_aware_messages.setdefault(user, []).append(query)
        for i, m in enumerate(["Bon", "Bonjour", "Bonjour " + query]):
            if p.lent:
                await asyncio.sleep(p.lent)
            yield m, (p.wav if i == 2 else None), "dit"


class MoteurGlobal:
    """Faux `Llm` global d'Aria. Les tours passent par le VRAI `Llm.get_answer_web` (c'est lui qui écrit le journal d'audit)."""

    def __init__(self):
        from components.llm import Llm
        self._llm = Llm.__new__(Llm)
        self._llm.backend_type = "anthropic"
        self.appels = []
        self.lent = 0.0
        self.wav = None
        self.engine = Moteur(self)
        self._llm.engine = self.engine

    @property
    def user_aware_messages(self):
        return self.engine.user_aware_messages

    def get_answer_web(self, tts, query, user, contexte=""):
        return self._llm.get_answer_web(tts, query, user, contexte)


class McpMemoire:
    """Faux serveur de mémoire : outil caché ui_etat (interlocuteur global)."""

    def __init__(self):
        self.actif = ""
        self.appels = []
        self.liste = [{"id": "p1", "label": "Olivier"}, {"id": "p2", "label": "Tiphaine"}]

    def has_tool(self, nom):
        return nom == "ui_etat"

    async def ensure_connected(self):
        pass

    async def close(self):
        pass

    async def call_tool(self, nom, args):
        self.appels.append((nom, dict(args)))
        activer = args.get("activer", "?")
        if activer != "?":
            self.actif = activer
        contexte = f"Interlocuteur actuel : {self.actif or 'Inconnue'}."
        return json.dumps({"liste": self.liste, "actif": self.actif, "contexte": contexte})


class Texte:
    @staticmethod
    def stt(valeur):
        class S:
            def transcribe_translate(self, samples):
                return valeur
        return S()


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class Base(unittest.TestCase):
    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.tmp = Path(tempfile.mkdtemp(prefix="chat_aria_"))
        self.config = copy.deepcopy(CONFIG)
        self.llm, self.mcp = MoteurGlobal(), McpMemoire()
        echanges.configure(self.tmp / "audit", enabled=True)
        echanges.clear()
        store = keys.KeyStore(self.tmp / "data")
        registre = sessions.SessionRegistry(profile.ConfigProfiles(lambda: self.config), lambda p, i=None: (None, None), max_sessions=5, ttl_s=900,
                                            shared_engine=lambda: (self.llm, self.mcp))
        from components.agents import api
        self.deps = api.Deps(keys=store, sessions=registre, stt=None, tts=object(), decode_audio=lambda raw: [0.0] * 16000, on_audit=lambda *a, **k: None,
                             static_dir=str(self.tmp), reset_on_close=True)
        self.deps.stt = Texte.stt("quelle heure est-il")
        self.agents = server.Agents(self.deps, store, None, None)
        self.chat = chat.ChatAria(self.agents, self.config, lambda: self.mcp, tts=object())

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        shutil.rmtree(self.tmp, ignore_errors=True)

    def lancer(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def tour(self, texte=None, audio=None, choisi="", **kw):
        async def g():
            return [e async for e in self.chat.tour(texte, audio, choisi, **kw)]
        return self.lancer(g())

    def noms(self, ev):
        return [n for n, _ in ev]

    def journal(self):
        return [l for _, p in echanges.files() for l in echanges.read_lines(p)]


class TourTests(Base):
    def test_un_tour_texte_renvoie_le_texte_cumule(self):
        ev = self.tour("bonjour")
        self.assertEqual([d for n, d in ev if n == "texte"], ["Bon", "Bonjour", "Bonjour bonjour"])

    def test_le_chat_utilise_le_moteur_global_et_la_session_aria_ui(self):
        self.tour("bonjour")
        self.assertEqual(self.llm.appels[0]["user"], "aria:ui")
        self.assertIn("aria:ui", self.llm.user_aware_messages)

    def test_le_contexte_d_interlocuteur_est_transmis_au_moteur(self):
        self.tour("salut", choisi="p1")
        self.assertEqual(self.llm.appels[0]["contexte"], "Interlocuteur actuel : p1.")

    def test_l_interlocuteur_choisi_est_communique_a_la_memoire_avant_le_tour(self):
        self.tour("salut", choisi="p2")
        self.assertEqual(self.mcp.actif, "p2")
        self.assertEqual(self.mcp.appels[0], ("ui_etat", {"activer": "p2"}))

    def test_le_nom_de_l_interlocuteur_est_ecrit_dans_le_journal_d_audit(self):
        self.tour("salut", choisi="p1")
        (ligne,) = self.journal()
        self.assertEqual((ligne["who"], ligne["src"]), ("Olivier", "aria:Aria"))

    def test_sans_interlocuteur_le_journal_dit_inconnue(self):
        self.tour("salut")
        self.assertEqual(self.journal()[0]["who"], "Inconnue")

    def test_le_journal_garde_la_conversation_de_la_session(self):
        self.tour("un")
        self.tour("deux")
        convs = {l["conv"] for l in self.journal()}
        self.assertEqual(convs, {"aria:ui#1"})

    def test_le_tour_vocal_donne_user_text(self):
        ev = self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertEqual(ev[0], ("user_text", "quelle heure est-il"))

    def test_le_chemin_du_wav_est_rendu_au_chat(self):
        self.llm.wav = "/tmp/bloc.wav"
        ev = self.tour("salut")
        self.assertEqual([d for n, d in ev if n == "audio"], [{"path": "/tmp/bloc.wav", "text": "dit"}])

    def test_sans_synthese_le_moteur_n_en_recoit_pas(self):
        self.chat.tts = None
        self.tour("salut")
        self.assertFalse(self.llm.appels[0]["tts"])

    def test_les_outils_deviennent_des_evenements(self):
        ligne = "\n\n> 🛠️ Outil utilisé : `meteo`\n\n"

        async def avec_outil(tts, query, user, contexte=""):
            yield "Je cherche" + ligne, None, None
            yield "Je cherche" + ligne + " Il pleut", None, None
        self.llm.engine.get_answer_web = avec_outil
        ev = self.tour("météo")
        self.assertEqual([d for n, d in ev if n == "outil"], ["meteo"])
        self.assertTrue(all("🛠️" not in d for n, d in ev if n == "texte"))

    def test_une_erreur_du_modele_devient_un_evenement_erreur(self):
        async def casse(tts, query, user, contexte=""):
            yield "Erreur LLM: quota", None, None
        self.llm.engine.get_answer_web = casse
        ev = self.tour("x")
        self.assertEqual(self.noms(ev)[-1], "erreur")


class ConversationTests(Base):
    def test_une_phrase_dite_hors_conversation_est_ignoree_avec_un_ordre_idle(self):
        self.deps.stt = Texte.stt("une phrase sans mot d'ouverture")
        ev = self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertEqual(ev[-1], ("ctl", "idle"))
        self.assertEqual(self.llm.appels, [])
        self.assertEqual(self.journal(), [])

    def test_le_mot_d_ouverture_ouvre_et_repond(self):
        self.deps.stt = Texte.stt("bonjour aria quelle heure")
        ev = self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertIn(("ctl", "conv_open"), ev)
        self.assertIn("texte", self.noms(ev))
        self.assertLess(ev.index(("ctl", "conv_open")), self.noms(ev).index("texte"))                 # l'icône apparaît avant la réponse

    def test_le_texte_saisi_part_toujours_sans_ouvrir(self):
        ev = self.tour("une question")
        self.assertIn("texte", self.noms(ev))
        self.assertNotIn(("ctl", "conv_open"), ev)

    def test_un_mot_d_arret_donne_un_ordre_stop_sans_appeler_le_modele(self):
        ev = self.tour("stop")
        self.assertEqual(ev[-1], ("ctl", "stop"))
        self.assertEqual(self.llm.appels, [])

    def test_la_cloture_oublie_l_interlocuteur_et_l_historique(self):
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA", choisi="p1")
        self.assertEqual(self.mcp.actif, "p1")
        self.deps.stt = Texte.stt("merci aria")
        ev = self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertEqual(ev[-1], ("ctl", "conv_close"))
        self.assertEqual(self.mcp.actif, "")                                                         # « Inconnue » : Aria ne sait plus à qui elle parlait
        self.assertNotIn("aria:ui", self.llm.user_aware_messages)                                    # historique du moteur effacé

    def test_la_cloture_envoie_bien_l_ordre_d_oubli_a_la_memoire(self):
        """On observe l'APPEL fait au serveur de mémoire, pas seulement son état final : l'ordre `activer=""` doit partir à la clôture."""
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA", choisi="p1")
        self.mcp.appels.clear()
        self.mcp.actif = "p1"
        self.deps.stt = Texte.stt("merci aria")
        self.tour(audio="data:audio/wav;base64,AAAA", choisi="p1")                                   # la liste montre toujours p1 : le tour ne l'efface donc pas lui-même
        ordres = [a["activer"] for n, a in self.mcp.appels if n == "ui_etat"]
        self.assertEqual(ordres[0], "p1")                                                            # avant le tour : l'interlocuteur choisi
        self.assertEqual(ordres[-1], "")                                                             # après la clôture : l'oubli, envoyé par la clôture et par elle seule
        self.assertEqual(self.mcp.actif, "")

    def test_la_cloture_garde_l_historique_si_reset_on_close_est_faux(self):
        self.config["memory"]["reset_on_close"] = False
        self.deps.reset_on_close = False
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA", choisi="p1")
        self.deps.stt = Texte.stt("merci aria")
        ev = self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertEqual(ev[-1], ("ctl", "conv_close"))
        self.assertIn("aria:ui", self.llm.user_aware_messages)
        self.assertEqual(self.mcp.actif, "")                                                         # mais l'interlocuteur, lui, est toujours oublié

    def test_la_cloture_change_de_conversation_pour_le_journal(self):
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA")
        self.deps.stt = Texte.stt("merci aria")
        self.tour(audio="data:audio/wav;base64,AAAA")
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA")
        self.assertEqual([l["conv"] for l in self.journal()], ["aria:ui#1", "aria:ui#1", "aria:ui#2"])


class InterlocuteurTests(Base):
    def test_rafraichir_suit_le_changement_decide_par_aria(self):
        self.mcp.actif = "p2"
        etat, dispo = self.lancer(self.chat.rafraichir_interlocuteur())
        self.assertEqual((etat["actif"], dispo, self.chat.dernier_interlocuteur), ("p2", True, "p2"))

    def test_passer_d_une_personne_a_une_autre_efface_le_contexte(self):
        self.tour("salut", choisi="p1")
        self.chat.dernier_interlocuteur = "p1"
        efface = self.lancer(self.chat.changer_interlocuteur("p2"))
        self.assertTrue(efface)
        self.assertNotIn("aria:ui", self.llm.user_aware_messages)
        self.assertEqual(self.mcp.actif, "p2")

    def test_passer_de_inconnue_a_une_personne_n_efface_rien(self):
        self.tour("salut")
        self.chat.dernier_interlocuteur = ""
        self.assertFalse(self.lancer(self.chat.changer_interlocuteur("p1")))
        self.assertIn("aria:ui", self.llm.user_aware_messages)

    def test_la_meme_personne_n_efface_rien(self):
        self.tour("salut", choisi="p1")
        self.chat.dernier_interlocuteur = "p1"
        self.assertFalse(self.lancer(self.chat.changer_interlocuteur("p1")))

    def test_le_changement_d_interlocuteur_change_de_conversation_d_audit(self):
        self.tour("un", choisi="p1")
        self.chat.dernier_interlocuteur = "p1"
        self.lancer(self.chat.changer_interlocuteur("p2"))
        self.tour("deux", choisi="p2")
        self.assertEqual([l["conv"] for l in self.journal()], ["aria:ui#1", "aria:ui#2"])

    def test_sans_serveur_de_memoire_tout_se_degrade_vers_inconnue(self):
        self.chat.mcp_manager = lambda: None
        ev = self.tour("salut")
        self.assertIn("texte", self.noms(ev))
        self.assertEqual(self.journal()[0]["who"], "Inconnue")
        etat, dispo = self.lancer(self.chat.rafraichir_interlocuteur())
        self.assertFalse(dispo)


class ArretEtNouvelleConversationTests(Base):
    def test_arreter_coupe_la_reponse_en_cours(self):
        self.llm.lent = 0.05

        async def g():
            sortie = []
            async for e in self.chat.tour("long"):
                sortie.append(e)
                if e[0] == "texte":
                    self.chat.arreter()
            return sortie
        ev = self.lancer(g())
        self.assertEqual(ev[-1], ("ctl", "stop"))

    def test_arreter_sans_session_ne_leve_rien(self):
        self.chat.arreter()

    def test_nouvelle_conversation_efface_ferme_et_oublie(self):
        self.deps.stt = Texte.stt("bonjour aria")
        self.tour(audio="data:audio/wav;base64,AAAA", choisi="p1")
        self.assertTrue(self.chat.conv.active)
        self.lancer(self.chat.nouvelle_conversation())
        self.assertFalse(self.chat.conv.active)
        self.assertNotIn("aria:ui", self.llm.user_aware_messages)
        self.assertEqual(self.mcp.actif, "")
        self.tour("après")
        self.assertEqual(self.journal()[-1]["conv"], "aria:ui#2")

    def test_nouvelle_conversation_coupe_la_reponse_en_cours(self):
        self.llm.lent = 0.05

        async def scenario():
            premier = asyncio.ensure_future(self._tour_complet("long"))
            await asyncio.sleep(0.07)
            await self.chat.nouvelle_conversation()
            return await premier
        ev = self.lancer(scenario())
        self.assertEqual(ev[-1], ("ctl", "stop"))

    async def _tour_complet(self, texte):
        return [e async for e in self.chat.tour(texte)]

    def test_nouvelle_conversation_avant_toute_session(self):
        self.lancer(self.chat.nouvelle_conversation())                                               # ne doit pas planter
        self.assertEqual(self.mcp.actif, "")

    def test_la_session_est_reutilisee_d_un_tour_a_l_autre(self):
        self.tour("un")
        self.tour("deux")
        self.assertEqual(len(self.deps.sessions.sessions), 1)

    def test_oublier_respecte_reset_on_close_sauf_force(self):
        self.tour("un")
        self.config["memory"]["reset_on_close"] = False
        self.assertFalse(self.lancer(self.chat.oublier()))
        self.assertIn("aria:ui", self.llm.user_aware_messages)
        self.assertTrue(self.lancer(self.chat.oublier(force=True)))
        self.assertNotIn("aria:ui", self.llm.user_aware_messages)


if __name__ == "__main__":
    unittest.main()
