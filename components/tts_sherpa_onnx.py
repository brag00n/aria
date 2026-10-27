import glob
import os
import logging
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
import sherpa_onnx
from components.utils import log_info, clean_text_for_tts, aria_path

logger = logging.getLogger("aria.tts.sherpa")

RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"


class TtsSherpaOnnx:
    """Voix Piper via sherpa-onnx. Le modèle (donc la langue) est choisi par l'appelant :
    model_name = nom de l'archive de la release "tts-models" (ex. "vits-piper-fr_FR-upmc-medium").
    Il est téléchargé dans models/sherpa/<model_name> s'il manque."""

    def __init__(self, params=None, device=None, model_name=None):
        self.params = params or {}
        self.base_models_dir = aria_path("models", "sherpa")
        self.model_name = model_name or "vits-piper-fr_FR-upmc-medium"

        # model_dir de la config : seulement s'il correspond au modèle demandé (sinon il pointerait vers une autre langue)
        configured = self.params.get("model_dir")
        if configured and os.path.basename(os.path.normpath(configured)) == self.model_name:
            self.model_dir = configured
        else:
            self.model_dir = os.path.join(self.base_models_dir, self.model_name)

        self.download_url = f"{RELEASE_URL}/{self.model_name}.tar.bz2"
        self.speed = self.params.get("speed", 0.8)
        self.noise_scale = self.params.get("noise_scale", 0.667)
        self.noise_scale_w = self.params.get("noise_scale_w", 0.8)

        self._ensure_model_exists()
        self._load_model()

    def _find_onnx(self):
        """Premier fichier .onnx du dossier du modèle (son nom varie selon la langue), ou None."""
        found = sorted(glob.glob(os.path.join(self.model_dir, "*.onnx")))
        return found[0] if found else None

    def _ensure_model_exists(self):
        """Vérifie la présence du modèle ou le télécharge."""
        if self._find_onnx():
            return
        log_info("Sherpa-ONNX", f"📥 Modèle {self.model_name} manquant. Téléchargement depuis GitHub...")
        os.makedirs(self.base_models_dir, exist_ok=True)
        archive_path = os.path.join(self.base_models_dir, f"{self.model_name}.tar.bz2")
        try:
            urllib.request.urlretrieve(self.download_url, archive_path)
            log_info("Sherpa-ONNX", "📦 Extraction de l'archive...")
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(path=self.base_models_dir)
            os.remove(archive_path)
        except Exception as e:
            log_info("Sherpa-ONNX", f"❌ ÉCHEC du téléchargement automatique : {e}")
            if os.path.exists(archive_path):
                os.remove(archive_path)          # une archive partielle ne doit pas rester
            raise
        if not self._find_onnx():
            raise FileNotFoundError(f"Aucun fichier .onnx dans {self.model_dir} après extraction")
        log_info("Sherpa-ONNX", "✅ Modèle installé avec succès.")

    def _load_model(self):
        onnx_path = self._find_onnx()
        log_info("Sherpa-ONNX", f"Chargement du moteur : {onnx_path}")
        tokens_path = os.path.join(self.model_dir, "tokens.txt")
        data_dir = os.path.join(self.model_dir, "espeak-ng-data")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=onnx_path,
                    tokens=tokens_path,
                    data_dir=data_dir,
                    noise_scale=self.noise_scale,
                    noise_scale_w=self.noise_scale_w
                ),
                num_threads=4,
                provider="cpu",
            )
        )

        if not tts_config.validate():
            raise ValueError("Erreur de validation Sherpa-ONNX.")

        self.model_obj = sherpa_onnx.OfflineTts(tts_config)
        log_info("Sherpa-ONNX", f"Voix prête ({self.model_name}).")

    def generate(self, text, output_path, *args, **kwargs):
        try:
            gen_text = clean_text_for_tts(text)
            audio_data = self.model_obj.generate(gen_text, sid=0, speed=self.speed)
            samples = audio_data.samples

            if len(samples) > 0 and np.abs(samples).max() > 0:
                samples = samples / np.abs(samples).max()

            sf.write(output_path, samples, audio_data.sample_rate)
            return output_path
        except Exception as e:
            log_info("Sherpa-ONNX", f"ERREUR : {str(e)}")
            return None
