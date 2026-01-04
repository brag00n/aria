import warnings
import components.utils as utils
from faster_whisper import WhisperModel

class Stt:
    def __init__(self, params=None):
        self.params = params or {}
        # On détecte si CUDA est demandé et disponible
        self.device = self.params.get("device", "cpu")
        self.model_name = "large-v3-turbo" 
        
        self.language = self.params.get("language", "fr")
        dest_models_dir = "/aria/models"

        utils.log_perf("Stt", f"Chargement Faster-Whisper ({self.model_name}) sur {self.device}...")
        
        self.model = WhisperModel(
            self.model_name, # "large-v3-turbo"
            device="cuda" if "cuda" in self.device else "cpu",
            device_index=0 if "cuda" in self.device else None,
            compute_type="float16" if "cuda" in self.device else "int8",
            download_root=dest_models_dir
        )

    def transcribe_translate(self, data):
        # beam_size=1 pour la vitesse maximale (temps réel)
        # vad_filter=True pour ignorer les silences et bruits de fond
        segments, info = self.model.transcribe(
            data, 
            beam_size=1, 
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Concaténation des segments de texte
        text = "".join([segment.text for segment in segments])
        return text.strip()