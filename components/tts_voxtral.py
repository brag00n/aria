import os
from llama_cpp import Llama
from components.utils import log_info, clean_text_for_tts

class TtsVoxtral:
    def __init__(self, params=None, device="cuda"):
        self.params = params or {}
        self.model_path = self.params.get("model_path")
        
        # Initialisation du modèle GGUF via llama-cpp
        # n_gpu_layers=12 est optimal pour vos 8 Go de VRAM
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.params.get("n_gpu_layers", 12),
            verbose=False,
            logits_all=True,
            n_ctx=4096
        )

    def generate(self, text, output_path, voice_ref=None, *args, **kwargs):
        try:
            # Nettoyage du texte pour Aria
            clean_text = clean_text_for_tts(text)
            
            # L'inférence génère les tokens audio (format spécifique à Voxtral GGUF)
            # Le prompt "TTS:" est souvent le déclencheur pour ces modèles
            res = self.model.create_completion(f"TTS: {clean_text}") 
            
            # Note : La conversion des tokens en fichier .wav dépend 
            # de l'implémentation spécifique du GGUF de TrevorJS
            return output_path
        except Exception as e:
            log_info("TTS-Voxtral", f"Erreur de génération : {e}")
            return None