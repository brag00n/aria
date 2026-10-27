import os
import sys
import time
import soundfile as sf
import components.utils as utils
from components.language import Language

class Tts:
    """Voix d'Aria. Chaîne de repli, avec un log à chaque étape :
    moteur demandé (Piper/Sherpa ou Kokoro) pour la langue globale -> l'autre moteur -> aucune voix (réponse écrite seule)."""

    def __init__(self, params=None, language=None):
        self.params = params or {}
        self.device = self.params.get("device", "cuda:0")
        # Par défaut, on privilégie désormais 'sherpa' pour l'identité d'Héléna
        self.tts_type = self.params.get("tts_type", "sherpa")
        self.tmp_dir = utils.aria_path("tmp", "tts")
        self.language = language or Language("fr")
        self.engine = None
        os.makedirs(self.tmp_dir, exist_ok=True)

        order = ["sherpa", "kokoro"] if self.tts_type == "sherpa" else ["kokoro", "sherpa"]
        utils.log_info("TTS", f"Langue '{self.language.code}' : ordre d'essai {' -> '.join(order)}")
        for name in order:
            engine = self._try_sherpa() if name == "sherpa" else self._try_kokoro()
            if engine is not None:
                self.engine, self.tts_type = engine, name
                utils.log_info("TTS", f"✅ Voix '{name}' active pour la langue '{self.language.code}'")
                break
        if self.engine is None:
            utils.log_info("TTS", f"❌ AUCUNE voix disponible pour la langue '{self.language.code}' : "
                                  "Aria répondra par écrit seulement")

    def _try_sherpa(self):
        """Voix Piper de la liste connue (téléchargée au besoin), ou None."""
        model = self.language.piper_model(self.params.get("piper_models"))
        if model is None:
            utils.log_info("TTS", f"Piper : pas de modèle pour '{self.language.code}' -> repli")
            return None
        try:
            from components.tts_sherpa_onnx import TtsSherpaOnnx
            return TtsSherpaOnnx(params=self.params, device=self.device, model_name=model)
        except Exception as e:
            utils.log_info("TTS", f"Piper : échec du modèle '{model}' ({e}) -> repli")
            return None

    def _try_kokoro(self):
        """Voix Kokoro pour la langue, ou None."""
        code = self.language.kokoro_code()
        if code is None:
            utils.log_info("TTS", f"Kokoro : langue '{self.language.code}' non utilisable -> repli")
            return None
        code, voice = self.language.kokoro_voice(code, self.params.get("kokoro_voice"))
        try:
            from components.tts_kokoro import TtsKokoro
            return TtsKokoro(params=self.params, device=self.device, lang_code=code, voice=voice)
        except Exception as e:
            utils.log_info("TTS", f"Kokoro : échec ({e}) -> repli")
            return None

    def run_tts_to_file(self, text, user_id="default"):
        """Génère un fichier audio unique via le moteur sélectionné (None si aucune voix n'est disponible)."""
        if self.engine is None:
            return None
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

                // Signale au visualiseur (static/audioDesign.js) qu'un bloc va être joué
                window.dispatchEvent(new CustomEvent('aria_audio_start', {{ detail: {{ audioElement: audio }} }}));

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