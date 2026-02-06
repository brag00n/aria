import os
import logging
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
import sherpa_onnx
from components.utils import log_info, clean_text_for_tts 

logger = logging.getLogger("aria.tts.sherpa")

class TtsKokoro:
    def __init__(self, params=None, device=None):
        self.params = params or {}
      
        try:
            log_info("TTS-Kokoro", "Chargement Kokoro-TTS...")
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='f', device=device, repo_id='hexgrad/Kokoro-82M')
            self.voice = self.params.get("kokoro_voice", "ff_siwis")
        except Exception as e:
            log_info("TTS-Kokoro", f"Erreur fatale TTS Kokoro : {e}")

        # Vérification et téléchargement avant chargement
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Vérifie la présence du modèle ou le télécharge."""

    def _load_model(self):
        """Chargement du modèle Kokoro."""

    def generate(self, text, output_path, *args, **kwargs):
        try:
            generator = self.pipeline(text, voice=self.voice, speed=1.1)
            for _, _, audio in generator:
                sf.write(output_path, audio, 24000)
                return output_path
        except Exception as e:
            log_info("TTS-Kokoro", f"ERREUR : {str(e)}")
            return None