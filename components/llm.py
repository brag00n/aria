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