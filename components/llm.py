import os
import sys
import time
import json
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import components.utils as utils
from .utils import clean_text_for_tts, remove_nonverbal_cues
from .mcp_manager import McpManager

class Llm:
    def __init__(self, params=None, all_config=None):
        self.params = params or {}
        self.model_name = self.params.get("model_name", None)
        self.model_file = self.params.get("model_file", None)
        self.device = self.params.get("device", "gpu")
        self.num_gpu_layers = self.params.get("num_gpu_layers", -1)
        self.chat_format = self.params.get("chat_format", "mistral-instruct")
        self.system_message = self.params.get("system_message", "")
        self.verbose = self.params.get("verbose", False)

        # Initialisation MCP sécurisée (évite le blocage si un serveur est HS)
        self.mcp = None
        if all_config and "Mcp" in all_config:
            try:
                self.mcp = McpManager(all_config["Mcp"]["params"])
                self.mcp.start()
            except Exception as e:
                utils.log_perf("LLM", f"Alerte: MCP n'a pas pu démarrer: {e}")

        # Téléchargement/Chargement du modèle GGUF
        dest_models_dir = "/aria/tmp/models"
        try:
            utils.log_perf("LLM", f"Téléchargement modèle {self.model_name} sur {dest_models_dir}...")
            model_path = hf_hub_download(repo_id=self.model_name, filename=self.model_file, local_dir=dest_models_dir)
        except Exception as e:
            utils.log_perf("LLM", f"Erreur téléchargement modèle: {e}")
            raise e

        # Initialisation de Llama avec gestion explicite du GPU
        main_gpu = 0 if "cuda" in self.device or self.device == "gpu" else None
        
        utils.log_perf("LLM", f"Chargement modele {self.model_name} sur {self.device}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=self.num_gpu_layers,
            n_ctx=4096,
            chat_format=self.chat_format,
            verbose=self.verbose,
            main_gpu=main_gpu
        )

        self.user_aware_messages = {}

    def get_answer_web(self, tts, query, user):
        """Génère une réponse en gérant les outils MCP et le streaming TTS."""

        start_time = time.time()

        if user not in self.user_aware_messages:
            self.user_aware_messages[user] = [{"role": "system", "content": self.system_message}]

        self.user_aware_messages[user].append({"role": "user", "content": query})
        
        # 1. ÉTAPE DE DÉCISION (TOOLS)
        # Correction CRITIQUE : S'assurer que 'tools' est une liste ou None, JAMAIS un booléen.
        tools = None
        if self.mcp:
            try:
                mcp_tools = self.mcp.get_tools_schema()
                # On ne garde que si c'est une liste non vide
                if isinstance(mcp_tools, list) and len(mcp_tools) > 0:
                    tools = mcp_tools
            except Exception as e:
                utils.log_perf("LLM", f"Erreur lecture schema outils: {e}")

        # Premier passage pour détecter si un outil doit être appelé
        try:
            # On construit les paramètres dynamiquement pour éviter d'envoyer tools=None ou tools=[]
            completion_params = {
                "messages": self.user_aware_messages[user],
                "temperature": 0.1
            }
            if tools:
                completion_params["tools"] = tools
                completion_params["tool_choice"] = "auto"

            response = self.llm.create_chat_completion(**completion_params)
        except Exception as e:
            utils.log_perf("LLM", f"Erreur completion (étape tools): {e}")
            yield "Désolée, j'ai rencontré une erreur technique.", None
            return

        message = response["choices"][0]["message"]

        # 2. EXÉCUTION DES OUTILS SI NÉCESSAIRE
        if "tool_calls" in message and message["tool_calls"] and self.mcp:
            self.user_aware_messages[user].append(message)
            
            utils.log_perf("LLM", f"   Debut Appel des outils MCP...")
            for tool_call in message["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"])
                except Exception:
                    fn_args = {}
                
                tool_result = self.mcp.call_tool_sync(fn_name, fn_args)
                
                self.user_aware_messages[user].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(tool_result)
                })
            utils.log_perf("LLM", f"   Fin Appel des outils MCP.")

        # 3. GÉNÉRATION DE LA RÉPONSE FINALE EN STREAMING
        try:
            outputs = self.llm.create_chat_completion(
                messages=self.user_aware_messages[user],
                stream=True,
                temperature=0.7
            )
        except Exception as e:
            utils.log_perf("LLM", f"Erreur completion (étape finale): {e}")
            yield "Mon cerveau vient de griller.", None
            return

        full_response = ""
        tts_text_buffer = ""
        
        token_count = 0
        for chunk in outputs:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                token = delta["content"]
                full_response += token
                tts_text_buffer += token
                token_count += 1 # On compte chaque morceau généré

                if any(punc in token for punc in [".", "!", "?", "\n"]):
                    txt_to_speak = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
                    if len(txt_to_speak) > 3:
                        audio_path = tts.run_tts_to_file(txt_to_speak, user_id=user)
                        tts_text_buffer = ""
                        yield full_response, audio_path
                    else:
                        yield full_response, None
                else:
                    yield full_response, None

        final_txt = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
        if len(final_txt) > 1:
            audio_path = tts.run_tts_to_file(final_txt, user_id=user)
            yield full_response, audio_path

        # Calcul des performances à la fin de la génération
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_sec = token_count / duration if duration > 0 else 0
        
        # Log final dans la console
        utils.log_perf("LLM", f"   Vitesse : {tokens_per_sec:.2f} t/s ({token_count} tokens en {duration:.2f}s)")


        self.user_aware_messages[user].append({"role": "assistant", "content": full_response})