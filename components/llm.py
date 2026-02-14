# components/llm.py
import components.utils as utils
from .llm_connect_local import LlmConnectLocal
from .llm_connect_openai import LlmConnectOpenai

class Llm:
    def __init__(self, params=None, all_config=None):
        self.params = params or {}
        self.backend_type = self.params.get("backend_type", "local")

        if self.backend_type in {"llm_studio", "ollama", "openai"}:
            self.engine = LlmConnectOpenai(self.params, all_config=all_config)
        else:
            utils.log_info("LLM", "Initialisation du mode Local 🏠")
            self.engine = LlmConnectLocal(self.params, all_config)

    async def get_answer_web(self, tts, query, user):
        async for values in self.engine.get_answer_web(tts, query, user):
            # Gestion flexible du nombre de valeurs (2 ou 3)
            response = values[0]
            audio_path = values[1]
            text_for_tts = values[2] if len(values) > 2 else ""
            
            yield response, audio_path, text_for_tts