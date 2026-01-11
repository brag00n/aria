import os
import sys
import time
import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import components.utils as utils
from .utils import clean_text_for_tts, remove_nonverbal_cues

class Llm:
    def __init__(self, params=None):
        self.params = params or {}
        self.custom_path = self.params.get("custom_path", "")
        self.model_name = self.params.get("model_name", None)
        self.model_file = self.params.get("model_file", None)
        self.device = self.params.get("device", "gpu")
        self.num_gpu_layers = self.params.get("num_gpu_layers", -1)
        self.context_length = self.params.get("context_length", 32768)
        self.chat_format = self.params.get("chat_format", "mistral-instruct")
        self.system_message = self.params.get("system_message", "")
        self.verbose = self.params.get("verbose", False)

        dest_models_dir = "/aria/tmp/models"

        # Gestion du chemin du modèle
        if self.custom_path != "":
            model_path = self.custom_path
        else:
            filename_to_download = self.model_file[0] if isinstance(self.model_file, list) else self.model_file
            model_path = hf_hub_download(
                repo_id=self.model_name, 
                filename=filename_to_download,
                local_dir=dest_models_dir,
                local_dir_use_symlinks=False
            )

        utils.log_perf("Llm", f"Chargement Modele LLM {os.path.basename(model_path)} sur {self.device}...")

        # --- ASTUCE : Silence total sur stderr pour masquer les logs internes de llama-cpp ---
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=self.num_gpu_layers,
                n_ctx=self.context_length,
                chat_format=self.chat_format,
                verbose=False, # Désactive les logs de performance natifs
            )
        finally:
            # On restaure le flux d'erreur standard immédiatement
            sys.stderr.close()
            sys.stderr = stderr

        self.user_aware_messages = {}

    def get_answer_web(self, tts, query, user):
        """Génère une réponse en streaming et déclenche le TTS par phrases complètes."""
        
        # Initialisation de l'historique utilisateur
        if user not in self.user_aware_messages or len(self.user_aware_messages[user]) == 0:
            self.user_aware_messages[user] = [{"role": "system", "content": self.system_message}]
            
        self.user_aware_messages[user].append({"role": "user", "content": query})

        # Gestion de la fenêtre contextuelle (10 derniers messages)
        if len(self.user_aware_messages[user]) > 10:
            self.user_aware_messages[user] = [self.user_aware_messages[user][0]] + self.user_aware_messages[user][-9:]

        start_time = time.time()
        token_count = 0
        full_response = ""
        tts_text_buffer = "" 
        
        # Lancement de la génération streamée
        outputs = self.llm.create_chat_completion(
            messages=self.user_aware_messages[user],
            stream=True,
            temperature=0.8,      
            repeat_penalty=1.2,  
            top_p=0.95,            
            max_tokens=150
        )

        for chunk in outputs:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                full_response += token
                tts_text_buffer += token
                token_count += 1
                
                # --- DETECTION DE FIN DE PHRASE AMELIOREE ---
                # On cherche la ponctuation finale pour envoyer un bloc cohérent au TTS
                if any(punc in token for punc in [".", "!", "?", "\n"]):
                    # Nettoyage via utils.py (qui doit préserver les accents !)
                    txt_to_speak = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
                    
                    # On évite de lancer le TTS sur des micro-segments ou de la ponctuation seule
                    if len(txt_to_speak) > 5:
                        last_audio_path = tts.run_tts_to_file(txt_to_speak, user_id=user)
                        tts_text_buffer = "" # Vidage du buffer après synthèse
                        yield full_response, last_audio_path
                    else:
                        yield full_response, None
                else:
                    yield full_response, None

        # Traitement du reliquat final (dernière phrase sans point par exemple)
        final_txt = clean_text_for_tts(remove_nonverbal_cues(tts_text_buffer)).strip()
        if len(final_txt) > 2:
            last_audio_path = tts.run_tts_to_file(final_txt, user_id=user)
            yield full_response, last_audio_path

        # Log de performance
        duration = time.time() - start_time
        if duration > 0:
            utils.log_perf("LLM", f"   Vitesse : {token_count / duration:.2f} t/s")
            
        self.user_aware_messages[user].append({"role": "assistant", "content": full_response})