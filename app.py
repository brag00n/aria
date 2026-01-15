import os
import warnings
import logging

# --- SILENCE AUX WARNINGS (AVANT TOUT IMPORT) ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["LLAMA_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*symlinks.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gradio as gr
import numpy as np
import base64, io, json, shutil, time, re
from datetime import datetime
from pydub import AudioSegment
from components.stt import Stt
from components.llm import Llm
from components.tts import Tts
import components.utils as utils

# --- CONFIGURATION ---
DEBUG_SAVE_WAV = False
TMP_DIR = "/aria/tmp/tts"

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR, mode=0o777, exist_ok=True)

# --- INIT COMPOSANTS ---
utils.log_perf("app", "--- DEBUT Chargement de la configuration et des modeles...")
with open("configs/default.json", "r") as f:
    config = json.load(f)

stt = Stt(config["Stt_Whisper"]["params"])
llm = Llm(config["Llm_Ministral"]["params"])
tts = Tts(config["Tts_Kokoro"]["params"])

utils.log_perf("app", "--- FIN Chargement de la configuration et des modeles.")

# --- JAVASCRIPT TOTAL CDN & LOGS ---
JS_COMBO = """
async () => {
    if (window.ariaInitialized) return;
    
    const loadScript = (src) => new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });

    try {
        console.log("[HMI] Chargement des dépendances via CDN...");
        await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");
        await loadScript("https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js");
        ort.env.logLevel = "error";
        
        window.ariaInitialized = true;
        
        // ... (Logique de queue audio inchangée) ...
        const logo = document.querySelector('#aria_logo');
        window.audioQueue = [];
        window.playedIds = new Set();
        window.isPlaying = false;

        async function playNext() {
            if (window.audioQueue.length === 0) { window.isPlaying = false; return; }
            window.isPlaying = true;
            const chunk = window.audioQueue.shift();
            console.log("[STREAM] Lecture du bloc : " + chunk.id);
            const audio = new Audio(chunk.data);
            audio.onended = playNext;
            audio.onerror = () => playNext();
            audio.play().catch(e => playNext());
        }

        const bridge = document.querySelector('#audio_url_bridge textarea');
        if (bridge) {
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            Object.defineProperty(bridge, 'value', {
                set: function(val) {
                    nativeSetter.call(this, val);
                    if (!val) return;
                    try {
                        const chunks = JSON.parse(val);
                        chunks.forEach(chunk => {
                            if (!window.playedIds.has(chunk.id)) {
                                window.playedIds.add(chunk.id);
                                window.audioQueue.push(chunk);
                            }
                        });
                        if (!window.isPlaying) playNext();
                    } catch(e) {}
                }
            });
        }

        // --- RESTAURATION DES PARAMÈTRES VAD ---
        const myVAD = await vad.MicVAD.new({
            modelURL: "/file=static/silero_v6.2/silero_vad.onnx",
            onSpeechStart: () => {
                console.log("[VAD] Parole détectée");
                if (logo) logo.classList.add('speaking');
            },
            onSpeechEnd: (audio) => {
                console.log("[VAD] Fin de parole (envoi)");
                if (logo) logo.classList.remove('speaking');
                
                const wavBuffer = vad.utils.encodeWAV(audio);
                const base64Audio = "data:audio/wav;base64," + vad.utils.arrayBufferToBase64(wavBuffer);
                
                const container = document.getElementById('audio_input_box');
                const textarea = container ? container.querySelector('textarea') : null;
                
                if (textarea) {
                    textarea.value = ""; 
                    textarea.value = base64Audio;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    setTimeout(() => {
                        const btn = document.getElementById('aria_trigger');
                        if (btn) btn.click();
                    }, 150);
                }
            },
            // SEUILS RESTAURÉS ICI :
            positiveSpeechThreshold: 0.8,
            negativeSpeechThreshold: 0.4,
            minSpeechFrames: 3,
            redemptionFrames: 20, // Optionnel: temps avant de couper après le dernier mot
        });
        
        myVAD.start();
        console.log("[VAD] Système prêt avec seuils personnalisés.");

    } catch (err) { 
        console.error("[HMI ERROR]", err); 
    }
}
"""

# --- LOGIQUE SERVEUR ---
def process_streaming_binaire(b64_audio, history):
    if not b64_audio: yield history, ""; return
    if history is None: history = []
    try:
        header, encoded = b64_audio.split(",", 1)
        audio_data = base64.b64decode(encoded)
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_data)).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
        
        text_user = stt.transcribe_translate(samples)
        history.append([text_user, ""])
        yield history, ""
        
        response_gen = llm.get_answer_web(tts, text_user, "DefaultUser")
        chunk_count = 0
        stream_payload = []
        
        for text_update, audio_chunk_path in response_gen:
            history[-1][1] = text_update
            if audio_chunk_path and os.path.exists(audio_chunk_path):
                chunk_count += 1
                unique_id = f"bloc_{time.time_ns()}_{chunk_count}"
                with open(audio_chunk_path, "rb") as f:
                    bin_data = f.read()
                    b64_data = f"data:audio/wav;base64,{base64.b64encode(bin_data).decode('utf-8')}"
                
                # Log serveur pour le suivi de l'envoi
                utils.log_perf("app", f"   [STREAM] Bloc #{chunk_count} envoyé")
                
                if not DEBUG_SAVE_WAV:
                    try: os.remove(audio_chunk_path)
                    except: pass
                
                stream_payload.append({"id": unique_id, "data": b64_data})
                yield history, json.dumps(stream_payload)
            else:
                yield history, json.dumps(stream_payload)
                
    except Exception as e:
        utils.log_perf("app", f"!!! ERREUR : {str(e)}")
        yield history, ""

# --- UI GRADIO ---
CSS = """
#aria_logo.speaking { 
    border: 4px solid #ff4b4b; 
    box-shadow: 0 0 20px #ff4b4b;
    transform: scale(1.1);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
}
"""

with gr.Blocks(css=CSS, title="Aria Voice") as ariaHmi:
    gr.Markdown("# 🎙️ Aria Voice System")
    gr.Image(type="filepath", value="static/transition.gif", height="100", elem_id="aria_logo")
    chatbot = gr.Chatbot(elem_id="aria_chatbot")
    audio_url_bridge = gr.Textbox(visible=False, elem_id="audio_url_bridge")
    
    with gr.Row():
        start_btn = gr.Button("🚀 ACTIVER MICRO & SON", variant="primary")
        
    audio_input = gr.Textbox(visible=False, elem_id="audio_input_box")
    trigger_btn = gr.Button("Trigger", visible=False, elem_id="aria_trigger")
    
    start_btn.click(None, None, None, js=JS_COMBO)
    
    trigger_btn.click(
        fn=process_streaming_binaire, 
        inputs=[audio_input, chatbot], 
        outputs=[chatbot, audio_url_bridge], 
        show_progress="hidden",
        queue=True
    )

if __name__ == "__main__":
    ariaHmi.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["static/"])