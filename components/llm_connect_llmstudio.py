# components/llm_connect_llmstudio.py
import asyncio
import components.utils as utils
from .utils import clean_text_for_tts, remove_nonverbal_cues

class LlmConnectLlmStudio:
    def __init__(self, params):
        # On n'importe openai qu'au moment où on crée le client

        try:
            from openai import AsyncOpenAI
        except ImportError:
            utils.log_perf("LLM-STUDIO", "❌ Erreur: Bibliothèque 'openai' manquante. Faites 'pip install openai'")
            raise

        self.params = params or {}
        self.base_url = self.params.get("base_url", "http://localhost:1234/v1")
        self.model_name = self.params.get("model_name", "local-model")
        self.system_message = self.params.get("system_message", "")
        
        utils.log_perf("LLM-LLMSTUDIO", f"Chargement modele {self.model_name} depuis LM Studio à {self.base_url}...")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="lm-studio")
        self.user_aware_messages = {}

    async def get_answer_web(self, tts, query, user):
        """Génère une réponse en streaming via l'API de LM Studio."""
        if user not in self.user_aware_messages:
            self.user_aware_messages[user] = [{"role": "system", "content": self.system_message}]

        self.user_aware_messages[user].append({"role": "user", "content": query})

        
        utils.log_perf("LLM-STUDIO", f"Appel API LM Studio pour l'utilisateur {user}...")
        try:
            # Appel API en mode streaming
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.user_aware_messages[user],
                stream=True,
                temperature=0.7
            )

            full_response = ""
            tts_text_buffer = ""

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    tts_text_buffer += token

                    # Détection de ponctuation pour envoyer au TTS (Héléna)
                    if any(punc in token for punc in [".", "!", "?", "\n"]):
                        txt_to_speak = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
                        if len(txt_to_speak) > 3:
                            # Génération de l'audio via le module TTS
                            audio_path = tts.run_tts_to_file(txt_to_speak, user_id=user)
                            tts_text_buffer = ""
                            yield full_response, audio_path
                        else:
                            yield full_response, None
                    else:
                        yield full_response, None

            # Fin de phrase restante
            final_txt = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
            if len(final_txt) > 1:
                audio_path = tts.run_tts_to_file(final_txt, user_id=user)
                yield full_response, audio_path

            self.user_aware_messages[user].append({"role": "assistant", "content": full_response})

        except Exception as e:
            utils.log_perf("LLM-STUDIO", f"❌ Erreur de connexion à LM Studio: {e}")
            yield "Désolée, je n'arrive pas à joindre LM Studio. Vérifie qu'il est bien lancé sur le port 1234.", None