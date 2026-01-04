import os
import warnings
import numpy as np
import components.utils as utils
from trainer.io import get_user_data_dir
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from kokoro import KPipeline

class Tts:
    def __init__(self, params=None, ap=None):
        self.params = params or {}
        # ... (vos assignations de variables restent identiques)
        self.device = self.params.get("device", None)
        self.tts_type = self.params.get("tts_type", None)
        self.model_name = self.params.get("model_name", None)
        self.kokoro_lang_code = self.params.get("kokoro_lang_code", None)
        # ...

        # Dossier de destination sur le volume monté (Windows)
        dest_models_dir = "/aria/models"

        if self.tts_type == "coqui":
            utils.log_perf("TTS", f"Chargement Modele TTS Coqui sur {self.device}...")
            if not self.verbose:
                warnings.filterwarnings("ignore", module="TTS")

            # Redéfinition du chemin du modèle pour pointer vers le volume externe
            self.model_path = os.path.join(
                dest_models_dir, "coqui", self.model_name.replace("/", "--")
            )
            
            if self.force_reload or not os.path.isdir(self.model_path):
                # On force ModelManager à télécharger dans notre dossier externe
                self.model_manager = ModelManager(output_prefix=os.path.join(dest_models_dir, "coqui"))
                self.model_path, _, _ = self.model_manager.download_model(self.model_name)

            self.config = XttsConfig()
            self.config.load_json(os.path.join(self.model_path, "config.json"))
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(
                self.config,
                checkpoint_dir=self.model_path,
                use_deepspeed=self.use_deepspeed,
            )
            if self.device == "gpu":
                self.model.cuda()

            self.gpt_cond_latent, self.speaker_embedding = (
                self.model.get_conditioning_latents(audio_path=[self.voice_to_clone])
            )

        elif self.tts_type == "kokoro":            
            utils.log_perf("TTS", f"Chargement Modele TTS Kokoro sur {self.device}...")
            
            os.environ["HF_HOME"] = dest_models_dir 
            self.pipeline = KPipeline(lang_code=self.kokoro_lang_code, device=self.device )