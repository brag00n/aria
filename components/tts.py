import os
import sys
import time
import soundfile as sf
import components.utils as utils

class Tts:
    def __init__(self, params=None):
        self.params = params or {}
        self.device = self.params.get("device", "cuda:0")
        # Par défaut, on privilégie désormais 'sherpa' pour l'identité d'Héléna
        self.tts_type = self.params.get("tts_type", "sherpa")
        self.tmp_dir = "/aria/tmp/tts"
        
        os.makedirs(self.tmp_dir, exist_ok=True)

        # --- MOTEUR SHERPA-ONNX (HÉLÉNA PRO) ---
        if self.tts_type == "sherpa":
            try:
                utils.log_info("TTS", "Initialisation Sherpa-ONNX (Héléna)...")
                from components.tts_sherpa_onnx import TtsSherpaOnnx
                self.engine = TtsSherpaOnnx(device=self.device)
            except Exception as e:
                utils.log_info("TTS", f"Échec Sherpa-ONNX : {e}. Repli sur Kokoro...")
                self.tts_type = "kokoro"

        # --- MOTEUR KOKORO (BACKUP STABLE) ---
        if self.tts_type == "kokoro":
            try:
                utils.log_info("TTS", "Chargement Kokoro-TTS...")
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code='f', device=self.device, repo_id='hexgrad/Kokoro-82M')
                self.voice = self.params.get("kokoro_voice", "ff_siwis")
            except Exception as e:
                utils.log_info("TTS", f"Erreur fatale TTS Kokoro : {e}")

    def run_tts_to_file(self, text, user_id="default"):
        """Génère un fichier audio unique via le moteur sélectionné."""
        file_path = os.path.join(self.tmp_dir, f"aria_res_{user_id}_{time.time_ns()}.wav")

        try:
            # Exécution Sherpa-ONNX
            if self.tts_type == "sherpa":
                result = self.engine.generate(text, file_path)
                if result: return result
                # Fallback interne si la génération échoue
                self.tts_type = "kokoro" 

            # Exécution F5-TTS
            if self.tts_type == "f5":
                result = self.engine.generate(text, self.ref_audio, self.ref_text, file_path)
                if result: return result

            # Exécution Kokoro (Dernier recours ou mode rapide)
            if self.tts_type == "kokoro":
                generator = self.pipeline(text, voice=self.voice, speed=1.1)
                for _, _, audio in generator:
                    sf.write(file_path, audio, 24000)
                    return file_path
                    
        except Exception as e:
            utils.log_info("TTS", f"Erreur run_tts_to_file ({self.tts_type}) : {e}")
        
        return None