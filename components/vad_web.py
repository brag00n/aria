# components/vad_web.py

class VadWeb:
    def __init__(self, params):
        self.params = params
        self.pos_threshold = self.params.get("positive_threshold", 0.8)
        self.neg_threshold = self.params.get("negative_threshold", 0.4)
        self.redemption_frames = self.params.get("redemption_frames", 20)

    def get_javascript(self):
        """Logique VAD 100% autonome et auto-exécutée."""
        return f"""
        // La promesse est exposée : le bouton micro attend que le VAD soit prêt (ou en échec) avant de changer d'état
        window.ariaVadPromise = (async () => {{

            // 1. Verrouillage pour éviter les exécutions multiples
            if (window.ariaInitialized) {{
                console.log("[ARIA-VAD] Système déjà initialisé.");
                if (window.myVAD) window.myVAD.start();
                return;
            }}

            const loadScript = (src) => new Promise((resolve, reject) => {{
                const script = document.createElement('script');
                script.src = src;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            }});

            try {{
                console.log("[ARIA-CLIENT] Initialisation autonome du VAD...");
                
                // 2. Chargement des dépendances
                if (typeof ort === 'undefined') {{
                    await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");
                }}
                if (typeof vad === 'undefined') {{
                    await loadScript("https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js");
                }}
                
                ort.env.logLevel = "error";

                // 3. Préparation des variables globales pour le streaming
                window.audioQueue = window.audioQueue || [];
                window.playedIds = window.playedIds || new Set();
                window.isPlaying = false;
                window.ariaInitialized = true;

                // 4. Configuration du VAD
                window.myVAD = await vad.MicVAD.new({{
                    modelURL: "/file=static/silero_v6.2/silero_vad.onnx",
                    onSpeechStart: () => {{
                        const visualizer = document.getElementById('audio_visualizer');
                        if (visualizer) visualizer.classList.add('speaking');
                    }},
                    onVADMisfire: () => {{
                        // Bruit trop court pour être une phrase : on retire le liseré rouge, sinon il reste allumé
                        const visualizer = document.getElementById('audio_visualizer');
                        if (visualizer) visualizer.classList.remove('speaking');
                    }},
                    onSpeechEnd: (audio) => {{
                        const visualizer = document.getElementById('audio_visualizer');
                        if (visualizer) visualizer.classList.remove('speaking');
                        // Phrase détectée : l'image passe à « chargement » jusqu'à la réponse d'Aria (components/mic_toggle.py)
                        if (window.ariaMic) window.ariaMic.onPhrase();

                        const wavBuffer = vad.utils.encodeWAV(audio);
                        const base64Audio = "data:audio/wav;base64," + vad.utils.arrayBufferToBase64(wavBuffer);
                        
                        const container = document.getElementById('audio_input_box');
                        const textarea = container ? container.querySelector('textarea') : null;
                        
                        if (textarea) {{
                            textarea.value = base64Audio;
                            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            setTimeout(() => {{
                                const btn = document.querySelector('#aria_trigger');
                                if (btn) btn.click();
                            }}, 50);
                        }}
                    }},
                    positiveSpeechThreshold: {self.pos_threshold},
                    negativeSpeechThreshold: {self.neg_threshold},
                    minSpeechFrames: 3,
                    redemptionFrames: {self.redemption_frames},
                }});
                
                await window.myVAD.start();
                console.log("[ARIA-VAD] Système prêt (Auto-invoked).");

            }} catch (e) {{
                console.error("[ARIA-CLIENT] Erreur d'initialisation VAD:", e);
                window.ariaInitialized = false;
            }}
        }})(); 
        """