"""Moteur local (components/llm_connect_local.py) avec un FAUX modèle : historique par utilisateur, contexte d'interlocuteur, outils asynchrones, synthèse vocale,
arrêt et erreur. Ni llama-cpp ni fichier .gguf. Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import asyncio
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from components.llm_connect_local import LlmConnectLocal
    HAVE = True
except ImportError:
    HAVE = False


def morceau(texte):
    return {"choices": [{"delta": {"content": texte}}]}


class FauxLlama:
    """Imite llama_cpp.Llama.create_chat_completion : réponse complète, ou flux de morceaux. `scenario` : liste de réponses, une par appel."""

    def __init__(self, scenario):
        self.scenario = list(scenario)
        self.appels = []
        self.en_cours = 0
        self.max_en_cours = 0
        self.fil_appel = []
        self._v = threading.Lock()

    def create_chat_completion(self, messages, stream=False, tools=None, tool_choice=None, **kw):
        with self._v:
            self.en_cours += 1
            self.max_en_cours = max(self.max_en_cours, self.en_cours)
        self.fil_appel.append(threading.current_thread().name)
        self.appels.append({"messages": json.loads(json.dumps(messages)), "stream": stream, "tools": tools, "kw": kw})
        time.sleep(0.01)
        try:
            etape = self.scenario.pop(0) if self.scenario else "…"
            if stream:
                texte = etape if isinstance(etape, str) else (etape.get("content") or "")
                return iter([morceau(texte[i:i + 4]) for i in range(0, len(texte), 4)])
            if isinstance(etape, dict) and etape.get("tool_calls"):
                return {"choices": [{"message": {"content": "", "tool_calls": etape["tool_calls"]}}]}
            return {"choices": [{"message": {"content": etape if isinstance(etape, str) else etape.get("content", "")}}]}
        finally:
            with self._v:
                self.en_cours -= 1


class FauxMcp:
    def __init__(self, outils=("meteo",), resultat="Il fait beau"):
        self.outils, self.resultat, self.appels = outils, resultat, []

    async def get_tools_for_openai(self):
        return [{"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object", "properties": {}}}} for n in self.outils]

    async def call_tool(self, nom, args):
        self.appels.append((nom, args))
        return self.resultat


class FauxTts:
    def __init__(self):
        self.textes = []

    def run_tts_to_file(self, texte, user_id="x"):
        self.textes.append((texte, user_id))
        return f"/tmp/faux_{len(self.textes)}.wav"


def appel_outil(nom="meteo", args=None, ident="call_123456789"):
    return {"tool_calls": [{"id": ident, "type": "function", "function": {"name": nom, "arguments": json.dumps(args or {})}}]}


def moteur(scenario, mcp=None, memoire=None):
    llama = FauxLlama(scenario)
    e = LlmConnectLocal({"system_message": "Tu es Aria."}, all_config={"memory": memoire} if memoire else None, llama_factory=lambda p: llama)
    if mcp is not None:
        e.mcp_manager = e.mcp = mcp
    return e, llama


async def tour(e, texte="Bonjour", user="u1", contexte="", tts=None):
    sortie = []
    async for full, audio, spoken in e.get_answer_web(tts, texte, user, contexte):
        sortie.append((full, audio, spoken))
    return sortie


def lancer(coro):
    return asyncio.run(coro)


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class LocalTests(unittest.TestCase):
    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    # ---- historique par utilisateur
    def test_reponse_simple_et_historique_conserve(self):
        e, llama = moteur(["Bonjour à vous."])
        s = lancer(tour(e, "Salut"))
        self.assertEqual(s[-1][0], "Bonjour à vous.")
        h = e.user_aware_messages["u1"]
        self.assertEqual([m["role"] for m in h], ["system", "user", "assistant"])
        self.assertEqual((h[1]["content"], h[2]["content"]), ("Salut", "Bonjour à vous."))

    def test_le_second_tour_voit_le_premier(self):
        e, llama = moteur(["Réponse un.", "Réponse deux."])
        lancer(tour(e, "Question un"))
        lancer(tour(e, "Question deux"))
        envoye = llama.appels[-1]["messages"]
        self.assertEqual([m["content"] for m in envoye if m["role"] == "user"], ["Question un", "Question deux"])
        self.assertIn("Réponse un.", [m["content"] for m in envoye if m["role"] == "assistant"])

    def test_deux_utilisateurs_deux_historiques(self):
        e, _ = moteur(["Pour A.", "Pour B."])
        lancer(tour(e, "Je suis A", user="sessionA"))
        lancer(tour(e, "Je suis B", user="sessionB"))
        self.assertEqual([m["content"] for m in e.user_aware_messages["sessionA"] if m["role"] == "user"], ["Je suis A"])
        self.assertEqual([m["content"] for m in e.user_aware_messages["sessionB"] if m["role"] == "user"], ["Je suis B"])

    def test_l_effacement_d_un_utilisateur_ne_touche_pas_l_autre(self):
        from components import history
        e, _ = moteur(["a", "b"])
        lancer(tour(e, "x", user="A"))
        lancer(tour(e, "y", user="B"))
        history.reset(e.user_aware_messages, "A")
        self.assertNotIn("A", e.user_aware_messages)
        self.assertIn("B", e.user_aware_messages)

    def test_historique_borne_par_la_memoire(self):
        e, llama = moteur(["r"] * 8, memoire={"max_exchanges": 3, "max_chars": 0})
        for i in range(8):
            lancer(tour(e, f"question {i}"))
        questions = [m["content"] for m in e.user_aware_messages["u1"] if m["role"] == "user"]
        self.assertLessEqual(len(questions), 3)
        self.assertEqual(questions[-1], "question 7")

    # ---- contexte d'interlocuteur
    def test_le_contexte_est_ajoute_au_message_systeme_sans_se_cumuler(self):
        e, llama = moteur(["a", "b", "c"])
        lancer(tour(e, "1", contexte="Interlocuteur actuel : Olivier."))
        lancer(tour(e, "2", contexte="Interlocuteur actuel : Tiphaine."))
        lancer(tour(e, "3"))
        systemes = [a["messages"][0]["content"] for a in llama.appels]
        self.assertIn("Olivier", systemes[0])
        self.assertIn("Tiphaine", systemes[1])
        self.assertNotIn("Olivier", systemes[1])                                # jamais cumulé
        self.assertEqual(systemes[2], "Tu es Aria.")                            # sans contexte : le message système d'origine

    # ---- outils asynchrones
    def test_un_outil_est_appele_puis_la_reponse_en_tient_compte(self):
        mcp = FauxMcp()
        e, llama = moteur([appel_outil("meteo", {"ville": "Lyon"}), "Il fait beau à Lyon."], mcp=mcp)
        s = lancer(tour(e, "Quel temps à Lyon ?"))
        self.assertEqual(mcp.appels, [("meteo", {"ville": "Lyon"})])
        self.assertIn("Il fait beau à Lyon.", s[-1][0])
        self.assertIn("Outil utilisé : `meteo`", s[-1][0])                      # ligne d'information dans le texte affiché
        roles = [m["role"] for m in e.user_aware_messages["u1"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        outil = next(m for m in e.user_aware_messages["u1"] if m["role"] == "tool")
        self.assertEqual((outil["name"], outil["content"]), ("meteo", "Il fait beau"))

    def test_l_identifiant_d_appel_est_complete_s_il_est_trop_court(self):
        e, _ = moteur([appel_outil("meteo", {}, ident="c1"), "ok"], mcp=FauxMcp())
        lancer(tour(e, "x"))
        assistant = next(m for m in e.user_aware_messages["u1"] if m.get("tool_calls"))
        self.assertGreaterEqual(len(assistant["tool_calls"][0]["id"]), 9)
        outil = next(m for m in e.user_aware_messages["u1"] if m["role"] == "tool")
        self.assertEqual(outil["tool_call_id"], assistant["tool_calls"][0]["id"])

    def test_arguments_illisibles_donnent_un_dictionnaire_vide(self):
        mcp = FauxMcp()
        e, _ = moteur([{"tool_calls": [{"id": "call_123456789", "function": {"name": "meteo", "arguments": "{pas du json"}}]}, "ok"], mcp=mcp)
        lancer(tour(e, "x"))
        self.assertEqual(mcp.appels, [("meteo", {})])

    def test_sans_gestionnaire_d_outils_l_erreur_est_renvoyee_au_modele(self):
        e, _ = moteur(["ok"])
        e.mcp_manager = e.mcp = None
        s = lancer(tour(e, "bonjour"))
        self.assertEqual(s[-1][0], "ok")

    def test_la_limite_de_tours_d_outils_force_une_reponse_sans_outil(self):
        mcp = FauxMcp()
        e, llama = moteur([appel_outil("meteo", ident=f"call_00000000{i}") for i in range(4)] + ["Réponse finale."], mcp=mcp)
        s = lancer(tour(e, "boucle"))
        self.assertIn("Réponse finale.", s[-1][0])
        self.assertEqual(len(mcp.appels), 4)
        self.assertTrue(llama.appels[-1]["stream"])                             # dernier tour : en flux, sans outils
        self.assertIsNone(llama.appels[-1]["tools"])

    def test_les_outils_sont_proposes_au_modele_quand_ils_existent(self):
        e, llama = moteur(["ok"], mcp=FauxMcp(("a", "b")))
        lancer(tour(e, "x"))
        self.assertEqual([t["function"]["name"] for t in llama.appels[0]["tools"]], ["a", "b"])

    # ---- synthèse vocale
    def test_la_reponse_est_synthetisee_par_blocs(self):
        e, _ = moteur(["Bonjour Olivier. Comment allez-vous ?"])
        tts = FauxTts()
        s = lancer(tour(e, "salut", tts=tts))
        self.assertTrue(tts.textes)
        self.assertTrue(any(audio for _, audio, _ in s))
        self.assertTrue(all(user == "u1" for _, user in tts.textes))
        self.assertIn("Bonjour Olivier", " ".join(t for t, _ in tts.textes))

    def test_sans_synthese_aucun_audio(self):
        e, _ = moteur(["Une phrase. Une autre."])
        s = lancer(tour(e, "x", tts=None))
        self.assertFalse(any(audio for _, audio, _ in s))

    def test_la_ligne_outil_n_est_jamais_lue(self):
        e, _ = moteur([appel_outil("meteo"), "Il fait beau."], mcp=FauxMcp())
        tts = FauxTts()
        lancer(tour(e, "x", tts=tts))
        self.assertNotIn("Outil", " ".join(t for t, _ in tts.textes))

    # ---- erreur et arrêt
    def test_une_erreur_du_modele_renvoie_un_message_et_annule_le_tour(self):
        e, llama = moteur(["ok"])

        def casse(**kw):
            raise RuntimeError("modèle indisponible")
        llama.create_chat_completion = casse
        s = lancer(tour(e, "x"))
        self.assertTrue(s[-1][0].startswith("Erreur LLM:"))
        self.assertEqual([m["role"] for m in e.user_aware_messages["u1"]], ["system"])      # historique cohérent : le tour est retiré

    def test_un_arret_en_cours_de_reponse_annule_le_tour(self):
        e, _ = moteur(["Une très longue réponse qui n'en finit pas de se dérouler."])

        async def interrompu():
            gen = e.get_answer_web(None, "x", "u1", "")
            await gen.__anext__()
            await gen.aclose()
        lancer(interrompu())
        self.assertEqual([m["role"] for m in e.user_aware_messages["u1"]], ["system"])

    def test_un_tour_sans_reponse_retire_aussi_la_question(self):
        e, _ = moteur([""])
        lancer(tour(e, "x"))
        self.assertEqual([m["role"] for m in e.user_aware_messages["u1"]], ["system"])

    # ---- thread et verrou (le modèle est synchrone)
    def test_le_modele_s_execute_hors_du_thread_de_la_boucle(self):
        """Les DEUX voies : la détection d'outil (sans flux) et la réponse en flux."""
        for mcp, scenario in ((None, ["ok en flux"]), (FauxMcp(), ["ok sans outil demandé"]), (FauxMcp(), [appel_outil("meteo"), "ok après outil"])):
            e, llama = moteur(scenario, mcp=mcp)
            lancer(tour(e, "x"))
            self.assertTrue(llama.fil_appel)
            self.assertNotIn(threading.current_thread().name, llama.fil_appel, (mcp, scenario))

    def test_un_seul_appel_du_modele_a_la_fois(self):
        e, llama = moteur(["a b c d e f g h i j", "k l m n o p q r s t", "u v w x y z"] * 2)

        async def trois():
            await asyncio.gather(*(tour(e, f"q{i}", user=f"u{i}") for i in range(4)))
        lancer(trois())
        self.assertEqual(llama.max_en_cours, 1)

    def test_la_boucle_reste_libre_pendant_la_generation(self):
        e, llama = moteur(["Une réponse assez longue pour durer plusieurs jetons de suite."])
        battements = []

        async def battement():
            for _ in range(200):
                battements.append(1)
                await asyncio.sleep(0)
                if len(battements) > 50:
                    break

        async def les_deux():
            await asyncio.gather(tour(e, "x"), battement())
        lancer(les_deux())
        self.assertGreater(len(battements), 5)

    # ---- compatibilité
    def test_les_reglages_d_echantillonnage_sont_transmis(self):
        llama = FauxLlama(["ok"])
        e = LlmConnectLocal({"temperature": 0.2, "repeat_penalty": 1.1}, llama_factory=lambda p: llama)
        lancer(tour(e, "x"))
        self.assertEqual((llama.appels[0]["kw"]["temperature"], llama.appels[0]["kw"]["repeat_penalty"]), (0.2, 1.1))

    def test_attributs_attendus_par_le_reste_d_aria(self):
        e, _ = moteur(["ok"])
        self.assertIsInstance(e.user_aware_messages, dict)                       # reset des sessions (SessionRegistry.reset) et forget_context
        self.assertEqual(e.backend_type, "local")

    def test_le_constructeur_cree_le_gestionnaire_d_outils_sous_ses_deux_noms(self):
        """Sans toucher aux attributs après coup : `mcp_manager` (lu par Llm, app.py, les sessions) et l'ancien nom `mcp` sont le MÊME objet."""
        llama = FauxLlama(["ok"])
        e = LlmConnectLocal({}, all_config={"mcp": {"params": {"servers": {}}}}, llama_factory=lambda p: llama)
        self.assertIsNotNone(e.mcp_manager)
        self.assertIs(e.mcp, e.mcp_manager)
        self.assertIsNone(LlmConnectLocal({}, llama_factory=lambda p: llama).mcp_manager)             # sans section mcp : pas d'outils

    def test_un_agent_avec_le_droit_obtient_un_moteur_local_par_build_engine(self):
        """Chemin réel des sessions : build_engine -> Llm -> LlmConnectLocal. Le modèle est remplacé par un faux (pas de fichier .gguf)."""
        import unittest.mock as mock
        from components.agents import profile, sessions
        llama = FauxLlama(["Réponse locale."])
        cfg = {"active_modules": {"llm": "Llm_Local"}, "llm": {"Llm_Local": {"params": {"backend_type": "local", "system_message": "Local."}}}}
        prof = {"name": "aria", "system_message": "", "llm": "", "mcp": {"servers": {}}, "app_prompt": "", "rights": {"local_llm": True}}
        with mock.patch.object(LlmConnectLocal, "_charger_modele", lambda self: llama):
            llm, mcp = sessions.build_engine(prof, cfg, None, ("aria", "Aria"))
            self.assertIsInstance(llm.engine, LlmConnectLocal)
            sortie = lancer(tour(llm.engine, "Bonjour"))
            self.assertEqual(sortie[-1][0], "Réponse locale.")
            with self.assertRaises(profile.ProfileError):                                        # le même profil SANS le droit reste refusé
                sessions.build_engine({**prof, "rights": {}}, cfg, None, ("k_1", "Robot"))

    def test_le_wrapper_llm_transmet_le_contexte_au_moteur_local(self):
        from components.llm import Llm
        llm = Llm.__new__(Llm)
        llm.engine, llm.backend_type = None, "local"
        e, llama = moteur(["ok"])
        llm.engine = e

        async def via_wrapper():
            return [x async for x in llm.get_answer_web(None, "x", "u1", "Interlocuteur actuel : Léa.")]
        lancer(via_wrapper())
        self.assertIn("Léa", llama.appels[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
