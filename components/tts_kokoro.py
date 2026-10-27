import logging
import soundfile as sf
from components.utils import log_info

logger = logging.getLogger("aria.tts.kokoro")

class TtsKokoro:
    """Voix Kokoro. lang_code (lettre Kokoro : "f" français, "a" anglais...) et voice sont choisis par l'appelant
    selon la langue globale (voir components/language.py) ; sans eux, la config est lue comme avant."""

    def __init__(self, params=None, device=None, lang_code=None, voice=None):
        self.params = params or {}
        self.device = device or self.params.get("device")
        self.voice = voice or self.params.get("kokoro_voice", "ff_siwis")
        self.speed = self.params.get("kokoro_voice_speed", 1.1)
        self.lang_code = lang_code or self.params.get("kokoro_lang_code", "f")
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Chargement du modèle Kokoro."""
        try:
            log_info("TTS-Kokoro", f"Chargement Kokoro-TTS (langue '{self.lang_code}', voix '{self.voice}')...")
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=self.lang_code, device=self.device, repo_id='hexgrad/Kokoro-82M')
        except Exception as e:
            log_info("TTS-Kokoro", f"Erreur fatale TTS Kokoro : {e}")
            raise

    def generate(self, text, output_path, *args, **kwargs):
        try:
            generator = self.pipeline(text, voice=self.voice, speed=self.speed)
            for _, _, audio in generator:
                sf.write(output_path, audio, 24000)
                return output_path
        except Exception as e:
            log_info("TTS-Kokoro", f"ERREUR : {str(e)}")
            return None
