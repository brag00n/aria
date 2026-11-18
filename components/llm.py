# components/llm.py
import components.utils as utils
from components import echanges
from .llm_connect_local import LlmConnectLocal
from .llm_connect_openai import LlmConnectOpenai

class Llm:
    def __init__(self, params=None, all_config=None, language=None):
        # Copie : on ne modifie pas le dict de la config, qui est réécrit tel quel par l'onglet Paramètres
        self.params = dict(params or {})
        self.params["system_message"] = (
            f"{self.params.get('system_message', '')}\n\n{utils.SPEECH_STYLE_PROMPT}".strip()
        )
        if language is not None:
            self.params["system_message"] += "\n\n" + language.llm_instruction()
            utils.log_info("LANG", f"LLM : consigne de langue '{language.code}' ajoutée au message système")
        self.backend_type = self.params.get("backend_type", "local")

        if self.backend_type == "anthropic":
            from .llm_connect_anthropic import LlmConnectAnthropic
            self.engine = LlmConnectAnthropic(self.params, all_config=all_config)
        elif self.backend_type in {"llm_studio", "ollama", "openai"}:
            self.engine = LlmConnectOpenai(self.params, all_config=all_config)
        else:
            utils.log_info("LLM", "Initialisation du mode Local 🏠")
            self.engine = LlmConnectLocal(self.params, all_config)

    async def get_answer_web(self, tts, query, user, contexte=""):
        # `contexte` (consigne propre à la requête, par exemple l'interlocuteur actuel) est géré par tous les moteurs, le moteur local compris
        args = (tts, query, user, contexte) if contexte else (tts, query, user)
        # Journal d'audit : UNE ligne par échange, écrite ici pour tous les moteurs (voir components/echanges.py). Le `finally` couvre aussi
        # l'arrêt (le générateur est fermé) et l'erreur. L'échange courant est posé par la source (interface ou agent) avant l'appel.
        echange = echanges.current()
        if echange is not None and not echange.query:
            echange.query = query or ""
        response, statut = "", "ok"
        try:
            async for values in self.engine.get_answer_web(*args):
                # Gestion flexible du nombre de valeurs (2 ou 3)
                response = values[0]
                audio_path = values[1]
                text_for_tts = values[2] if len(values) > 2 else ""

                yield response, audio_path, text_for_tts
            if statut == "ok" and not echanges.clean_answer(response):
                # Le modèle a terminé son tour (finish_reason=stop) sans écrire un mot : l'utilisateur resterait devant un silence, et le journal noterait « ok ».
                # C'est le cas d'un petit modèle qui s'embrouille dans un long message système ou une longue liste d'outils.
                modele = getattr(self.engine, "model_name", "?")
                utils.log_info("LLM", f"⚠️ Réponse vide du modèle {modele} ({self.backend_type}) : aucun texte"
                                      f"{', outils appelés : ' + ', '.join(echange.outils) if echange is not None and echange.outils else ''}. "
                                      "Raccourcir le message système ou la liste d'outils du profil, ou changer de modèle.")
                response = utils.NO_ANSWER_ERROR                      # le client ne voit AUCUN détail technique : tout est dans la ligne de log ci-dessus
                yield response, None, ""
        except GeneratorExit:
            statut = "interrompu"
            raise
        except BaseException:
            statut = "erreur"
            raise
        finally:
            if echange is not None:
                if statut == "ok" and str(response or "").startswith("Erreur LLM:"):
                    statut = "erreur"
                echanges.write(echange, response, statut)