import components.utils as utils
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from .utils import remove_emojis
from .utils import remove_nonverbal_cues
import time


class Llm:
    def __init__(self, params=None):
        self.params = params or {}
        self.custom_path = self.params.get("custom_path", "") # Correction ici pour gérer les chaînes vides
        self.model_name = self.params.get("model_name", None)
        self.model_file = self.params.get("model_file", None)
        self.device = self.params.get("device", None)
        self.num_gpu_layers = self.params.get("num_gpu_layers", None)
        self.context_length = self.params.get("context_length", None)
        self.streaming_output = self.params.get("streaming_output", None)
        self.chat_format = self.params.get("chat_format", None)
        self.system_message = self.params.get("system_message", None)
        self.verbose = self.params.get("verbose", None)

        # Définition du dossier de destination sur le volume monté
        # Cela correspond à C:\dev\Perso\prj_ai\aria\models sur votre Windows
        dest_models_dir = "/aria/models"

        if self.custom_path != "":
            model_path = self.custom_path
        else:
            # On utilise local_dir pour forcer le téléchargement dans le volume monté
            # local_dir_use_symlinks=False est important pour éviter les liens symboliques cassés
            
            filename_to_download = self.model_file[0] if isinstance(self.model_file, list) else self.model_file
            
            model_path = hf_hub_download(
                repo_id=self.model_name, 
                filename=filename_to_download,
                local_dir=dest_models_dir,
                local_dir_use_symlinks=False
            )

        utils.log_perf("Llm", f"Chargement modele LLM {model_path} sur {self.device}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=self.num_gpu_layers,
            n_ctx=self.context_length,
            chat_format=self.chat_format,
            verbose=self.verbose,
        )

        self.user_aware_messages = {}

# Dans components/llm_server.py

    def get_answer_web(self, tts, query, user):
        """Version pour Gradio / Web avec calcul de performance"""
        self.user_aware_messages[user].append({"role": "user", "content": query})
        
        start_time = time.time()
        token_count = 0
        full_response = ""
        tts_text_buffer = []
        
        # Appel LLM avec streaming
        outputs = self.llm.create_chat_completion(
            messages=self.user_aware_messages[user],
            stream=True
        )

        for chunk in outputs:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                full_response += token
                token_count += 1 # On compte chaque morceau généré
                
                yield full_response, None 

                # Logique TTS existante
                if any(punc in token for punc in [".", "!", "?", "\n"]):
                    raw_text = "".join(tts_text_buffer).strip()
                    txt_for_tts = remove_nonverbal_cues(remove_emojis(raw_text))
                    
                    if len(txt_for_tts) > 1:
                        # Ici, on génère l'audio mais on le gère différemment pour le Web
                        # Pour un premier test, on peut juste yield le texte
                        pass 
                    tts_text_buffer = []

        # Calcul des performances à la fin de la génération
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_sec = token_count / duration if duration > 0 else 0
        
        # Log final dans la console
        utils.log_perf("LLM", f"   Réponse totale : {full_response}")
        utils.log_perf("LLM", f"   Vitesse : {tokens_per_sec:.2f} t/s ({token_count} tokens en {duration:.2f}s)")

        self.user_aware_messages[user].append({"role": "assistant", "content": full_response})

    def get_answer(self, nw, tts, data, user):
        #self.messages.append({"role": "user", "content": data})
        if user not in self.user_aware_messages:
            self.user_aware_messages[user] = [{"role": "system", "content": self.system_message}]
        self.user_aware_messages[user].append({"role": "user", "content": data})

        outputs = self.llm.create_chat_completion(
            # self.messages, stream=self.streaming_output
            self.user_aware_messages[user], stream=self.streaming_output
        )

        if self.streaming_output:
            llm_output = ""
            tts_text_buffer = []
            color_code_block = False
            backticks = 0
            skip_code_block_on_tts = False
            nw.send_ack()
            for i, out in enumerate(outputs):
                if "content" in out["choices"][0]["delta"]:
                    output_chunk_txt = out["choices"][0]["delta"]["content"]
                    if (
                        output_chunk_txt == "```" and backticks == 0
                    ):  # 3 for qwen, 2 for mixtral ?
                        skip_code_block_on_tts = not skip_code_block_on_tts
                        color_code_block = not color_code_block
                        backticks += 3  # 3 for qwen, 2 for mixtral ?
                    if backticks > 0 and backticks <= 3:
                        backticks += 1
                    else:
                        backticks = 0
                    if i == 1:
                        if backticks == 0:
                            nw.send_msg("llm")
                            nw.send_msg(output_chunk_txt.strip())
                            nw.send_msg(str(color_code_block))
                    else:
                        if backticks == 0:
                            nw.send_msg("llm")
                            nw.send_msg(output_chunk_txt)
                            nw.send_msg(str(color_code_block))
                    llm_output += output_chunk_txt
                    if (
                        not skip_code_block_on_tts
                    ):  # fix when codeblock, tts not skiping leading text
                        tts_text_buffer.append(output_chunk_txt)
                        if tts_text_buffer[-1] in [".", "!", "?", ":", "..", "..."]:
                            # TODO handle float numbers
                            # TODO remove multi dots
                            # TODO handle emphasis
                            txt_for_tts = remove_nonverbal_cues(
                                    remove_emojis(
                                    "".join(tts_text_buffer).strip()
                                    )
                                )
                            # TODO fix 1 character
                            if len(txt_for_tts) > 1:
                                nw.send_msg("tts")
                                tts.run_tts(nw, txt_for_tts)
                            tts_text_buffer = []
            if not skip_code_block_on_tts and len(tts_text_buffer) != 0:
                # TODO remove multi dots
                txt_for_tts = remove_nonverbal_cues(remove_emojis("".join(tts_text_buffer).strip()))
                # TODO fix 1 character
                if len(txt_for_tts) > 1:
                    nw.send_msg("tts")
                    tts.run_tts(nw, txt_for_tts)
            nw.send_msg("streaming_end")
            llm_output = llm_output.strip()
        else:
            llm_output = outputs["choices"][0]["message"]["content"].strip()

        #self.messages.append({"role": "assistant", "content": llm_output})
        self.user_aware_messages[user].append({"role": "assistant", "content": llm_output})

        return llm_output
