# components/llm_connect_local.py
import os
import time
import json
import asyncio
import components.utils as utils
from llama_cpp import Llama
from .utils import clean_text_for_tts, remove_nonverbal_cues

class LlmConnectLocal:
    def __init__(self, params, all_config=None):
        self.params = params or {}
        self.system_message = self.params.get("system_message", "")
        self.chat_format = self.params.get("chat_format", "mistral-instruct")
        self.context_length = self.params.get("context_length", 4096)
        self.verbose = self.params.get("verbose", False)
        
        model_file = self.params.get("model_file", "ministral-3b-instruct.Q4_K_M.gguf")
        model_path = os.path.join("/aria/models/llm", model_file)
        
        utils.log_info("LLM-LOCAL", f"Chargement du modèle : {model_file}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            chat_format=self.chat_format,
            n_ctx=self.context_length,
            verbose=self.verbose
        )

        from .mcp import Mcp
        self.mcp = None
        if all_config and "Mcp" in all_config:
            self.mcp = Mcp(all_config["Mcp"]["params"])
            self.mcp.start()

    async def get_answer_web(self, tts, query, user):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query}
        ]

        # ÉTAPE 1 : Détection d'outil (Appel non-streamé pour analyser la structure)
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=self.mcp.get_tools_schema() if self.mcp else None,
            tool_choice="auto" 
        )

        choice = response["choices"][0]["message"]

        # ÉTAPE 2 : Si le modèle demande un outil
        if "tool_calls" in choice and choice["tool_calls"]:
            for tool_call in choice["tool_calls"]:
                t_name = tool_call["function"]["name"]
                t_args = json.loads(tool_call["function"]["arguments"])
                
                utils.log_info("LLM-LOCAL", f"🛠️ APPEL OUTIL : {t_name}({t_args})")
                
                # Exécution synchrone via MCP
                result = self.mcp.call_tool_sync(t_name, t_args)
                utils.log_info("MCP", f"✅ RÉPONSE : {result}")

                # On construit l'historique complet pour la réponse finale
                messages.append(choice)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": t_name,
                    "content": result
                })

            # ÉTAPE 3 : Génération finale avec les vraies données
            final_stream = self.llm.create_chat_completion(messages=messages, stream=True)
            async for chunk in self._stream_and_tts(final_stream, tts, user):
                yield chunk
        else:
            # Réponse directe si aucun outil n'est requis
            direct_stream = self.llm.create_chat_completion(messages=messages, stream=True)
            async for chunk in self._stream_and_tts(direct_stream, tts, user):
                yield chunk

    async def _stream_and_tts(self, stream, tts, user):
        full_text = ""
        buffer = ""
        for chunk in stream:
            await asyncio.sleep(0)
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                full_text += token
                buffer += token
                
                if any(p in token for p in [".", "!", "?", "\n"]):
                    txt = clean_text_for_tts(remove_nonverbal_cues(buffer)).strip()
                    if len(txt) > 2:
                        yield full_text, tts.run_tts_to_file(txt, user_id=user)
                        buffer = ""
                    else:
                        yield full_text, None
                else:
                    yield full_text, None
        
        if buffer.strip():
            yield full_text, tts.run_tts_to_file(buffer, user_id=user)