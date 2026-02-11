import torch
import components.utils as utils
from faster_whisper import WhisperModel

class SttFasterWhisper:
    def __init__(self, params):
        self.params = params
        # Utiliser un modèle déjà converti pour CT2
        original_model = self.params.get("model_name", "large-v3-turbo")
        
        # Mapping automatique si on oublie de préciser le suffixe CT2
        if "turbo" in original_model and "/" not in original_model:
            self.model_name = "deepdml/whisper-large-v3-turbo-ct2"
        elif "/" not in original_model:
            # Pour base, small, medium, etc. faster-whisper les gère nativement
            self.model_name = original_model 
        else:
            self.model_name = original_model

        raw_device = self.params.get("device", "cuda")
        self.device = "cuda" if "cuda" in raw_device.lower() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        utils.log_info("STT-FW", f"Chargement de {self.model_name} sur {self.device}")
        
        self.model = WhisperModel(
            self.model_name, 
            device=self.device, 
            compute_type=compute_type,
            # Supprimez download_root ou mettez un chemin simple pour laisser 
            # faster-whisper gérer son cache proprement
            device_index=0 if self.device == "cuda" else None
        )

    def transcribe_translate(self, audio_path):
        # beam_size=1 pour la vitesse, 5 pour la précision
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=1, 
            language="fr"
        )
        
        # Faster-Whisper renvoie un itérateur de segments
        full_text = " ".join([segment.text for segment in segments])
        return full_text.strip()