import os
import sys
import warnings
import time
import components.utils as utils
from kokoro import KPipeline
import soundfile as sf

class Tts:
    def __init__(self, params=None, ap=None):
        self.params = params or {}
        
        # Initialisation prioritaire
        self.verbose = self.params.get("verbose", False)
        self.device = self.params.get("device", "cpu")
        self.tts_type = self.params.get("tts_type", "kokoro")
        self.model_name = self.params.get("model_name", None)
        self.force_reload = self.params.get("force_reload", False)
        
        self.voice_to_use = self.params.get("kokoro_voice", "ff_siwis")
        self.voice_speed = self.params.get("kokoro_voice_speed", 1.0)
        self.kokoro_lang_code = self.params.get("kokoro_lang_code", "f")
        
        static_params = self.params.get("static", {})
        self.voice_to_clone = static_params.get("voice_to_clone", "static/sofia_hellen.wav")

        dest_models_dir = "/aria/tmp/models"

        if self.tts_type == "coqui":
            utils.log_perf("TTS", f"Chargement Modele TTS Coqui sur {self.device}...")
            
            # --- IMPORTS TARDIFS (LAZY IMPORTS) ---
            # Permet de démarrer Aria avec Kokoro même si Coqui/Transformers sont cassés
            try:
                from TTS.utils.manage import ModelManager
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
            except ImportError as e:
                utils.log_perf("TTS", f"!!! ERREUR CRITIQUE COQUI : Incompatibilité Transformers. Utilisez Kokoro. {e}")
                raise e

            if not self.verbose:
                import transformers
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", module="TTS")

            self.model_path = os.path.join(
                dest_models_dir, "coqui", self.model_name.replace("/", "--")
            )

            if self.force_reload or not os.path.isdir(self.model_path):
                utils.log_perf("TTS", f"   Téléchargement du modèle Coqui...")
                self.model_manager = ModelManager(output_prefix=os.path.join(dest_models_dir, "coqui"))
                self.model_path, _, _ = self.model_manager.download_model(self.model_name)

            self.config = XttsConfig()
            self.config.load_json(os.path.join(self.model_path, "config.json"))
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(
                self.config,
                checkpoint_dir=self.model_path,
                use_deepspeed=self.params.get("use_deepspeed", False),
            )
            
            if "cuda" in self.device or self.device == "gpu":
                self.model.cuda()

            utils.log_perf("TTS", "   Calcul de l'empreinte vocale (clonage)...")
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[self.voice_to_clone]
            )

        elif self.tts_type == "kokoro":         
            utils.log_perf("TTS", f"Chargement Modele TTS Kokoro sur {self.device}...")
            os.environ["HF_HOME"] = dest_models_dir
            # Capture de stderr pour silence
            stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                self.pipeline = KPipeline(lang_code=self.kokoro_lang_code, device=self.device, repo_id='hexgrad/Kokoro-82M')
            finally:
                sys.stderr.close()
                sys.stderr = stderr

    def run_tts_to_file(self, text, user_id="default"):
        output_dir = "/aria/tmp/tts"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"aria_res_{user_id}_{time.time_ns()}.wav")

        try:
            if self.tts_type == "kokoro":
                generator = self.pipeline(text, voice=self.voice_to_use, speed=self.voice_speed, split_pattern=r'\n+')
                for _, _, audio in generator:
                    sf.write(file_path, audio, 24000)
                    break

            elif self.tts_type == "coqui":
                out = self.model.inference(
                    text,
                    "fr", 
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                    temperature=0.7,
                )
                sf.write(file_path, out["wav"], 24000)

            return file_path
            
        except Exception as e:
            utils.log_perf("TTS", f"Erreur synthèse ({self.tts_type}) : {str(e)}")
            return None