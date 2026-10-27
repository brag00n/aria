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
        # Précision des calculs : float16 sur GPU, int8 sur CPU par défaut.
        # int8_float16 réduit la VRAM (medium : ~1,5-2 Go au lieu de ~2,5-3 Go) pour une qualité proche.
        compute_type = self.params.get("compute_type") or ("float16" if self.device == "cuda" else "int8")
        device_index = self.params.get("device_index", 0)

        # beam_size=1 pour la vitesse, 5 pour la précision
        self.beam_size = self.params.get("beam_size", 1)
        # Ignore les silences : évite les hallucinations ("Merci d'avoir regardé...") et gagne du temps
        self.vad_filter = self.params.get("vad_filter", True)
        self.min_silence_ms = self.params.get("min_silence_duration_ms", 500)
        self.condition_on_previous_text = self.params.get("condition_on_previous_text", False)

        utils.log_info("STT-FW", f"Chargement de {self.model_name} sur {self.device} ({compute_type})")

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=compute_type,
            # Pas de download_root : faster-whisper gère son cache proprement
            device_index=device_index
        )

    def transcribe_translate(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.params.get("language"),   # None = détection automatique
            vad_filter=self.vad_filter,
            vad_parameters={"min_silence_duration_ms": self.min_silence_ms},
            condition_on_previous_text=self.condition_on_previous_text,
        )
        
        # Faster-Whisper renvoie un itérateur de segments
        full_text = " ".join([segment.text for segment in segments])
        return full_text.strip()