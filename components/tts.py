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
                utils.log_info("TTS", "Initialisation Sherpa-ONNX...")
                from components.tts_kokoro import TtsKokoro
                self.engine = TtsKokoro(device=self.device)
            except Exception as e:
                utils.log_info("TTS", f"Échec Kokoro : {e}")
                raise RuntimeError("Aucun moteur TTS disponible.")

    def run_tts_to_file(self, text, user_id="default"):
        """Génère un fichier audio unique via le moteur sélectionné."""
        file_path = os.path.join(self.tmp_dir, f"aria_res_{user_id}_{time.time_ns()}.wav")

        try:
            # Exécution TTX configuré
            result = self.engine.generate(text, file_path)
            if result: return result
                    
        except Exception as e:
            utils.log_info("TTS", f"Erreur run_tts_to_file ({self.tts_type}) : {e}")
        
        return None