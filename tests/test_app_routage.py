"""Aiguillage du chat de l'interface (app.py) entre l'ancien chemin et l'agent par défaut `aria`, sans lancer Gradio ni charger les modèles : on extrait du fichier les
fonctions de la couche d'aiguillage (`chat_via_agent`, `_agent_tour`, `route_*`, `agent_*`) et on les exécute dans un espace de noms contrôlé.
Nécessite les dépendances d'Aria (conteneur) : ignoré sinon."""
import ast
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import components.utils as utils
    from components.agents import compat
    HAVE = True
except ImportError:
    HAVE = False

NOMS = ("chat_via_agent", "_chat", "_agent_tour", "agent_process_streaming_binaire", "agent_process_texte", "agent_new_conversation", "agent_stop",
        "agent_refresh_interlocutor", "agent_change_interlocutor", "route_process_streaming_binaire", "route_process_texte", "route_new_conversation",
        "route_stop_aria", "route_refresh_interlocutor", "route_change_interlocutor")


def extraire(espace):
    """Compile uniquement les fonctions de la couche d'aiguillage d'`app.py` dans `espace`."""
    arbre = ast.parse((ROOT / "app.py").read_text("utf-8"))
    morceaux = [n for n in arbre.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in NOMS]
    manquants = set(NOMS) - {n.name for n in morceaux}
    if manquants:
        raise AssertionError(f"fonctions absentes d'app.py : {sorted(manquants)}")
    exec(compile(ast.Module(body=morceaux, type_ignores=[]), "app.py", "exec"), espace)
    return espace


class FauxGr:
    @staticmethod
    def update(**kw):
        return kw


class FauxChat:
    def __init__(self, evenements):
        self.evenements, self.appels = evenements, []

    async def tour(self, texte=None, audio=None, interlocuteur_choisi="", speak=True):
        self.appels.append(("tour", texte, audio, interlocuteur_choisi))
        for e in self.evenements:
            yield e

    async def nouvelle_conversation(self):
        self.appels.append(("nouvelle",))

    def arreter(self):
        self.appels.append(("arreter",))

    async def rafraichir_interlocuteur(self):
        self.appels.append(("rafraichir",))
        return {"choices": [("Inconnue", ""), ("Olivier", "p1")], "actif": "p1"}, True

    async def changer_interlocuteur(self, choix):
        self.appels.append(("changer", choix))


@unittest.skipUnless(HAVE, "dépendances d'Aria absentes de l'hôte (lancer dans le conteneur)")
class Base(unittest.TestCase):
    def setUp(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.ancien = []
        self.tmp = tempfile.mkdtemp(prefix="routage_")

        def trace(nom, valeur=None):
            def f(*a, **k):
                self.ancien.append(nom)
                return valeur
            return f

        async def gen_voix(b64, history, interlocutor=""):
            self.ancien.append("process_streaming_binaire")
            yield history or [], "ancien-voix"

        async def gen_texte(text, history, interlocutor=""):
            self.ancien.append("process_texte")
            yield history or [], "ancien-texte", ""

        async def nouvelle():
            self.ancien.append("new_conversation")
            return [], "ancien-nouvelle"

        async def rafraichir():
            self.ancien.append("refresh_interlocutor")
            return {"ancien": True}

        async def changer(choix):
            self.ancien.append("change_interlocutor")

        self.config = {}
        self.espace = {"gr": FauxGr, "os": os, "time": time, "json": json, "base64": base64, "utils": utils, "agents_compat": compat,
                       "config": self.config, "chat_aria": None,
                       "ctl_payload": lambda kind: json.dumps([{"ctl": kind}]),
                       "process_streaming_binaire": gen_voix, "process_texte": gen_texte, "new_conversation": nouvelle,
                       "stop_aria": trace("stop_aria", "ancien-stop"), "refresh_interlocutor": rafraichir, "change_interlocutor": changer}
        extraire(self.espace)

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())

    def actif(self, evenements):
        self.config["agents"] = {"chat_via_agent": True}
        self.espace["chat_aria"] = FauxChat(evenements)
        return self.espace["chat_aria"]

    def attendre(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def collecter(self, gen):
        async def g():
            return [x async for x in gen]
        return self.attendre(g())


class AiguillageTests(Base):
    def test_par_defaut_l_ancien_chemin(self):
        self.assertFalse(self.espace["chat_via_agent"]())
        self.assertEqual(self.collecter(self.espace["route_process_texte"]("bonjour", [])), [([], "ancien-texte", "")])
        self.collecter(self.espace["route_process_streaming_binaire"]("data:...", []))
        self.attendre(self.espace["route_new_conversation"]())
        self.espace["route_stop_aria"]()
        self.attendre(self.espace["route_refresh_interlocutor"]())
        self.attendre(self.espace["route_change_interlocutor"]("p1"))
        self.assertEqual(self.ancien, ["process_texte", "process_streaming_binaire", "new_conversation", "stop_aria", "refresh_interlocutor", "change_interlocutor"])

    def test_le_drapeau_sans_chat_cree_reste_sur_l_ancien_chemin(self):
        self.config["agents"] = {"chat_via_agent": True}
        self.assertFalse(self.espace["chat_via_agent"]())                          # `chat_aria` n'existe pas : jamais de bascule dans le vide
        self.collecter(self.espace["route_process_texte"]("x", []))
        self.assertEqual(self.ancien, ["process_texte"])

    def test_le_drapeau_est_relu_a_chaque_appel(self):
        chat = self.actif([("texte", "ok")])
        self.collecter(self.espace["route_process_texte"]("un", []))
        self.config["agents"]["chat_via_agent"] = False                            # retour arrière à chaud, sans redémarrer
        self.collecter(self.espace["route_process_texte"]("deux", []))
        self.assertEqual([a[0] for a in chat.appels], ["tour"])
        self.assertEqual(self.ancien, ["process_texte"])

    def test_la_valeur_du_drapeau_doit_etre_vraie(self):
        self.actif([])
        for faux in (False, 0, None, "", []):
            self.config["agents"]["chat_via_agent"] = faux
            self.assertFalse(self.espace["chat_via_agent"](), repr(faux))
        for vrai in (True, 1, "oui"):
            self.config["agents"]["chat_via_agent"] = vrai
            self.assertTrue(self.espace["chat_via_agent"](), repr(vrai))

    def test_l_ancienne_section_access_est_aussi_lue(self):
        self.espace["chat_aria"] = FauxChat([])
        self.config["access"] = {"chat_via_agent": True}
        self.assertTrue(self.espace["chat_via_agent"]())

    def test_avec_le_drapeau_toutes_les_fonctions_passent_par_l_agent(self):
        chat = self.actif([("texte", "Bonjour")])
        self.collecter(self.espace["route_process_texte"]("salut", [], "p1"))
        self.collecter(self.espace["route_process_streaming_binaire"]("data:audio", [], "p1"))
        self.attendre(self.espace["route_new_conversation"]())
        self.assertEqual(self.espace["route_stop_aria"](), json.dumps([{"ctl": "stop"}]))
        self.attendre(self.espace["route_refresh_interlocutor"]())
        self.attendre(self.espace["route_change_interlocutor"]("p2"))
        self.assertEqual([a[0] for a in chat.appels], ["tour", "tour", "nouvelle", "arreter", "rafraichir", "changer"])
        self.assertEqual(self.ancien, [])                                           # l'ancien chemin n'a servi à rien


class TourAgentTests(Base):
    def test_le_texte_saisi_affiche_la_question_puis_la_reponse_cumulee(self):
        self.actif([("texte", "Bon"), ("texte", "Bonjour")])
        sortie = self.collecter(self.espace["route_process_texte"]("salut", []))
        histo = sortie[-1][0]
        self.assertEqual(histo, [{"role": "user", "content": "salut"}, {"role": "assistant", "content": "Bonjour"}])
        self.assertTrue(all(len(x) == 3 and x[2] == "" for x in sortie))              # champ de saisie vidé à chaque mise à jour

    def test_la_voix_affiche_ce_que_le_stt_a_compris(self):
        self.actif([("user_text", "quelle heure"), ("texte", "Midi")])
        sortie = self.collecter(self.espace["route_process_streaming_binaire"]("data:audio", []))
        self.assertEqual(sortie[-1][0], [{"role": "user", "content": "quelle heure"}, {"role": "assistant", "content": "Midi"}])

    def test_une_phrase_ignoree_n_ajoute_rien_au_chat(self):
        self.actif([("ctl", "idle")])
        sortie = self.collecter(self.espace["route_process_streaming_binaire"]("data:audio", []))
        self.assertEqual(sortie[-1][0], [])
        self.assertEqual(json.loads(sortie[-1][1]), [{"ctl": "idle"}])

    def test_le_son_est_lu_encode_puis_supprime(self):
        wav = os.path.join(self.tmp, "bloc.wav")
        Path(wav).write_bytes(b"RIFFfake")
        self.actif([("texte", "Salut"), ("audio", {"path": wav, "text": "Salut"})])
        sortie = self.collecter(self.espace["route_process_texte"]("x", []))
        charges = [json.loads(p) for _, p, _ in sortie if p and p.startswith("[")]
        (bloc,) = [c for c in charges if c and "data" in c[0]]
        self.assertEqual(base64.b64decode(bloc[0]["data"].split(",", 1)[1]), b"RIFFfake")
        self.assertTrue(bloc[0]["id"].startswith("bloc_"))
        self.assertFalse(os.path.exists(wav))

    def test_un_wav_introuvable_est_ignore(self):
        self.actif([("texte", "Salut"), ("audio", {"path": os.path.join(self.tmp, "absent.wav"), "text": "x"})])
        sortie = self.collecter(self.espace["route_process_texte"]("x", []))
        self.assertTrue(sortie)
        charges = [p for _, p, _ in sortie if p]
        self.assertEqual(charges, [])                                                 # aucun bloc audio émis pour un fichier absent, et aucune exception
        # Sans l'enveloppe de `route_process_texte` (qui avale les erreurs) : le garde lui-même doit écarter le bloc, pas une exception rattrapée plus haut.
        direct = self.collecter(self.espace["_agent_tour"]([], texte="x"))
        self.assertTrue(direct)
        self.assertEqual([p for _, p in direct if p], [])

    def test_les_ordres_pour_le_navigateur_passent_dans_le_pont_audio(self):
        self.actif([("ctl", "conv_open"), ("texte", "Oui"), ("ctl", "conv_close")])
        sortie = self.collecter(self.espace["route_process_streaming_binaire"]("data:audio", []))
        ordres = [json.loads(p)[0]["ctl"] for _, p in sortie if p and "ctl" in p]
        self.assertEqual(ordres, ["conv_open", "conv_close"])

    def test_une_erreur_s_affiche_dans_le_chat(self):
        self.actif([("erreur", "le modèle n'a pas pu répondre")])
        sortie = self.collecter(self.espace["route_process_texte"]("x", []))
        self.assertEqual(sortie[-1][0][-1]["content"], "Erreur : le modèle n'a pas pu répondre")

    def test_un_arret_immediat_ne_laisse_pas_les_points_de_suspension(self):
        self.actif([("texte", "..."), ("ctl", "stop")])
        sortie = self.collecter(self.espace["route_process_texte"]("x", []))
        self.assertNotEqual(sortie[-1][0][-1]["content"], "...")

    def test_l_interlocuteur_choisi_est_transmis(self):
        chat = self.actif([("texte", "ok")])
        self.collecter(self.espace["route_process_texte"]("x", [], "p2"))
        self.assertEqual(chat.appels[0], ("tour", "x", None, "p2"))

    def test_texte_vide_et_audio_vide(self):
        chat = self.actif([("texte", "ok")])
        self.assertEqual(self.collecter(self.espace["route_process_texte"]("   ", None)), [([], "", "")])
        self.assertEqual(self.collecter(self.espace["route_process_streaming_binaire"]("", None)), [([], "")])
        self.assertEqual(chat.appels, [])

    def test_une_exception_du_chat_ne_plante_pas_l_interface(self):
        class Casse(FauxChat):
            async def tour(self, *a, **k):
                raise RuntimeError("panne")
                yield
        self.config["agents"] = {"chat_via_agent": True}
        self.espace["chat_aria"] = Casse([])
        sortie = self.collecter(self.espace["route_process_texte"]("x", []))
        self.assertEqual(sortie[-1][1:], ("", ""))
        sortie = self.collecter(self.espace["route_process_streaming_binaire"]("data", []))
        self.assertEqual(sortie[-1][1], "")

    def test_la_liste_d_interlocuteurs_est_rafraichie_depuis_l_agent(self):
        self.actif([])
        maj = self.attendre(self.espace["route_refresh_interlocutor"]())
        self.assertEqual((maj["value"], maj["visible"]), ("p1", True))
        self.assertEqual(maj["choices"], [("Inconnue", ""), ("Olivier", "p1")])


class CablageTests(unittest.TestCase):
    """app.py : plus aucun composant Gradio ne contourne l'aiguillage."""

    def test_chaque_evenement_passe_par_une_fonction_route(self):
        import re
        src = (ROOT / "app.py").read_text("utf-8")
        directs = re.findall(r"(?:fn=|\.then\(|\.input\(|\.load\()(process_texte|process_streaming_binaire|new_conversation|stop_aria|refresh_interlocutor|change_interlocutor)\b", src)
        self.assertEqual(directs, [])
        for nom in ("route_process_texte", "route_process_streaming_binaire", "route_new_conversation", "route_stop_aria", "route_refresh_interlocutor",
                    "route_change_interlocutor"):
            self.assertIn(nom, src)

    def test_l_ancien_chemin_est_conserve(self):
        src = (ROOT / "app.py").read_text("utf-8")
        for nom in ("async def process_texte", "async def process_streaming_binaire", "async def _generer", "def forget_context", "async def new_conversation"):
            self.assertIn(nom, src)

    def test_le_drapeau_est_off_dans_la_config_par_defaut(self):
        cfg = json.loads((ROOT / "configs" / "default.json").read_text("utf-8"))
        for nom in ("agents", "access"):
            self.assertFalse((cfg.get(nom) or {}).get("chat_via_agent", False))


if __name__ == "__main__":
    unittest.main()
