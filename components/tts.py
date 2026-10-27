import os
import sys
import time
import soundfile as sf
from huggingface_hub import hf_hub_download
import components.utils as utils

class Tts:
    def __init__(self, params=None):
        self.params = params or {}
        self.device = self.params.get("device", "cuda:0")
        self.tts_type = self.params.get("tts_type", "sherpa")
        self.tmp_dir = "/aria/tmp/tts"
        
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.engine = None 


        # --- MOTEUR VOXTRAL (LLAMA-CPP GGUF) ---
        if self.tts_type == "voxtral":
            try:
                # 1. Gestion du téléchargement automatique du GGUF
                repo_id = self.params.get("repo_id")
                model_file = self.params.get("model_file")
                model_path = self.params.get("model_path")

                if repo_id and model_file and not os.path.exists(model_path):
                    utils.log_info("TTS", f"📥 Téléchargement du modèle Voxtral GGUF : {model_file}...")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=model_file,
                        local_dir=os.path.dirname(model_path),
                        local_dir_use_symlinks=False
                    )

                # 2. Instanciation de la classe TtsVoxtral
                from components.tts_voxtral import TtsVoxtral
                self.engine = TtsVoxtral(params=self.params, device=self.device)
                utils.log_info("TTS", "✅ Moteur TtsVoxtral (Llama-cpp) initialisé.")
            except Exception as e:
                utils.log_info("TTS", f"❌ Échec Voxtral : {e}. Repli sur Sherpa...")
                self.tts_type = "sherpa"

        # --- MOTEUR SHERPA-ONNX (FALLBACK) ---
        if self.tts_type == "sherpa" and self.engine is None:
            try:
                from components.tts_sherpa_onnx import TtsSherpaOnnx
                self.engine = TtsSherpaOnnx(device=self.device)
            except Exception as e:
                utils.log_info("TTS", f"Échec Sherpa-ONNX : {e}. Repli sur Kokoro...")
                self.tts_type = "kokoro"

        # --- MOTEUR KOKORO (BACKUP) ---
        if self.tts_type == "kokoro" and self.engine is None:
            try:
                from components.tts_kokoro import TtsKokoro
                self.engine = TtsKokoro(device=self.device)
            except Exception as e:
                utils.log_info("TTS", f"Échec Kokoro : {e}")
                raise RuntimeError("Aucun moteur TTS disponible.")

    def run_tts_to_file(self, text, user_id="default"):
        """Génère un fichier audio unique."""
        if not self.engine:
            return None

        file_path = os.path.join(self.tmp_dir, f"aria_res_{user_id}_{time.time_ns()}.wav")
        voice_ref = self.params.get("static", {}).get("voice_to_clone")

        try:
            if self.tts_type == "voxtral":
                # Appel de votre fonction dans tts_voxtral.py
                audio_data = self.engine(text, voice_ref)
                if audio_data:
                    with open(file_path, "wb") as f:
                        f.write(audio_data)
                    return file_path
            else:
                # Appel standard pour les autres moteurs (objets avec méthode .generate)
                return self.engine.generate(text, file_path)
                    
        except Exception as e:
            utils.log_info("TTS", f"Erreur run_tts_to_file ({self.tts_type}) : {e}")
        
        return None
    
    def get_javascript(self):
        return f"""
        try {{
            // 1. Initialisation des variables globales pour le streaming audio
            window.audioQueue = [];
            window.playedIds = new Set();
            window.isPlaying = false;

            // 2. Logique de lecture audio (Streaming)
            async function playNext() {{
                if (window.audioQueue.length === 0) {{ 
                    window.isPlaying = false; 
                    return; 
                }}
                window.isPlaying = true;
                const chunk = window.audioQueue.shift();
                console.log(`[ARIA-CLIENT] Lecture bloc audio: ${{chunk.id}}`);
                
                const audio = new Audio(chunk.data);

                // --- DÉCLENCHEMENT DE L'ANIMATION POUR LE VISUALISEUR ---
                // On émet l'événement que audioDesign.js écoute pour lier le flux
                const event = new CustomEvent('aria_audio_start', {{ 
                    detail: {{ audioElement: audio }} 
                }});
                window.dispatchEvent(event);
                // -------------------------------------------------------

                audio.onended = playNext;
                audio.onerror = (e) => {{
                    console.error(`[ARIA-CLIENT] Erreur lecture bloc ${{chunk.id}}:`, e);
                    playNext();
                }};

                audio.play().catch(e => {{
                    console.warn("[ARIA-CLIENT] Lecture bloquée par le navigateur (cliquez sur l'interface) :", e);
                    playNext();
                }});
            }}

            // 3. Surveillance du bridge pour les réponses TTS
            const bridge = document.querySelector('#audio_url_bridge textarea');
            if (bridge) {{
                const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                Object.defineProperty(bridge, 'value', {{
                    set: function(val) {{
                        nativeSetter.call(this, val);
                        if (!val) return;
                        try {{
                            const chunks = JSON.parse(val);
                            console.log(`[ARIA-CLIENT] Réception de ${{chunks.length}} bloc(s) audio.`);
                            chunks.forEach(chunk => {{
                                if (!window.playedIds.has(chunk.id)) {{
                                    window.playedIds.add(chunk.id);
                                    window.audioQueue.push(chunk);
                                }}
                            }});
                            if (!window.isPlaying) playNext();
                        }} catch(e) {{
                            console.error("[ARIA-CLIENT] Erreur parsing bridge:", e);
                        }}
                    }}
                }});
            }}

            console.log("[ARIA-CLIENT] Système prêt (VAD & Streaming Actifs).");
        }} catch (err) {{ 
            console.error("[ARIA-ERROR] Initialisation échouée:", err); 
        }}
        """