# components/llm.py
import components.utils as utils
from .llm_connect_local import LlmConnectLocal
from .llm_connect_llmstudio import LlmConnectLlmStudio

class Llm:
    def __init__(self, params=None, all_config=None):
        self.params = params or {}
        self.backend_type = self.params.get("backend_type", "local")

        if self.backend_type == "llm_studio":
            utils.log_perf("LLM", "Initialisation du mode LM Studio 🌐")
            # PASSAGE DE all_config ICI
            self.engine = LlmConnectLlmStudio(self.params, all_config=all_config)
        else:
            utils.log_perf("LLM", "Initialisation du mode Local 🏠")
            self.engine = LlmConnectLocal(self.params, all_config)

    async def get_answer_web(self, tts, query, user):
        async for response, audio_path in self.engine.get_answer_web(tts, query, user):
            yield response, audio_path