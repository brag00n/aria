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
                from components.tts_sherpa_onnx import TtsSherpaOnnx
                self.engine = TtsSherpaOnnx(device=self.device)
            except Exception as e:
                utils.log_info("TTS", f"Échec Sherpa-ONNX : {e}. Repli sur Kokoro...")
                self.tts_type = "kokoro"

        # --- MOTEUR KOKORO (BACKUP STABLE) ---
        if self.tts_type == "kokoro":
            try:
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
    
    def get_javascript(self):
        return f"""

        try {{
            // 2. Initialisation des variables globales pour le streaming audio
            window.audioQueue = [];
            window.playedIds = new Set();
            window.isPlaying = false;


            // 4. Logique de lecture audio (Streaming)
            async function playNext() {{
                if (window.audioQueue.length === 0) {{ 
                    window.isPlaying = false; 
                    return; 
                }}
                window.isPlaying = true;
                const chunk = window.audioQueue.shift();
                console.log(`[ARIA-CLIENT] Lecture bloc audio: ${{chunk.id}}`);
                const audio = new Audio(chunk.data);
                audio.onended = playNext;
                audio.onerror = (e) => {{
                    console.error(`[ARIA-CLIENT] Erreur lecture bloc ${{chunk.id}}:`, e);
                    playNext();
                }};
                audio.play().catch(e => {{
                    console.warn("[ARIA-CLIENT] Lecture bloquée par le navigateur:", e);
                    playNext();
                }});
            }}

            // 5. Surveillance du bridge pour les réponses TTS
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