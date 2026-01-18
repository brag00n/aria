import os
import logging
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
import sherpa_onnx
from components.utils import log_perf, clean_text_for_tts 

logger = logging.getLogger("aria.tts.sherpa")

class TtsSherpaOnnx:
    def __init__(self, params=None, device=None):
        self.params = params or {}
        
        # Structure : /aria/models/sherpa/<nom_du_modele>
        self.base_models_dir = "/aria/models/sherpa"
        self.model_name = "vits-piper-fr_FR-upmc-medium"
        self.model_dir = self.params.get("model_dir", os.path.join(self.base_models_dir, self.model_name))
        
        # URL de téléchargement officiel de Sherpa-ONNX
        self.download_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{self.model_name}.tar.bz2"
        
        self.speed = self.params.get("speed", 0.8)
        self.noise_scale = self.params.get("noise_scale", 0.667)
        self.noise_scale_w = self.params.get("noise_scale_w", 0.8)
        
        # Vérification et téléchargement avant chargement
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Vérifie la présence du modèle ou le télécharge."""
        onnx_path = os.path.join(self.model_dir, "fr_FR-upmc-medium.onnx")
        
        if not os.path.exists(onnx_path):
            log_perf("Sherpa-ONNX", f"📥 Modèle manquant. Téléchargement depuis GitHub...")
            os.makedirs(self.base_models_dir, exist_ok=True)
            
            archive_path = os.path.join(self.base_models_dir, f"{self.model_name}.tar.bz2")
            
            try:
                # Téléchargement
                urllib.request.urlretrieve(self.download_url, archive_path)
                log_perf("Sherpa-ONNX", "📦 Extraction de l'archive...")
                
                # Extraction
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(path=self.base_models_dir)
                
                # Nettoyage
                os.remove(archive_path)
                log_perf("Sherpa-ONNX", "✅ Modèle installé avec succès.")
                
            except Exception as e:
                log_perf("Sherpa-ONNX", f"❌ ÉCHEC du téléchargement automatique : {e}")
                raise

    def _load_model(self):
        log_perf("Sherpa-ONNX", f"Chargement du moteur : {self.model_dir}")
        
        onnx_path = os.path.join(self.model_dir, "fr_FR-upmc-medium.onnx")
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
        log_perf("Sherpa-ONNX", "Identité Héléna prête.")

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
            log_perf("Sherpa-ONNX", f"ERREUR : {str(e)}")
            return None