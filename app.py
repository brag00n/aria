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
utils.log_info("app", "--- DEBUT Chargement de la configuration et des modeles...")
with open("configs/default.json", "r") as f:
    config = json.load(f)

# On force l'usage de Ministral
stt = Stt(config["Stt_Whisper"]["params"])
llm = Llm(config["Llm_Ministral_ollama"]["params"], all_config=config)
tts = Tts(config["Tts_Kokoro"]["params"])
utils.log_info("app", "--- FIN Chargement de la configuration et des modeles.")

# --- JAVASCRIPT ---
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
        console.log("[ARIA-CLIENT] Initialisation du système...");
        await loadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js");
        await loadScript("https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js");
        ort.env.logLevel = "error";
        
        window.ariaInitialized = true;
        window.audioQueue = [];
        window.playedIds = new Set();
        window.isPlaying = false;

        async function playNext() {
            if (window.audioQueue.length === 0) { 
                window.isPlaying = false; 
                return; 
            }
            window.isPlaying = true;
            const chunk = window.audioQueue.shift();
            console.log(`[ARIA-CLIENT] Lecture bloc audio: ${chunk.id}`);
            const audio = new Audio(chunk.data);
            audio.onended = playNext;
            audio.onerror = (e) => {
                console.error(`[ARIA-CLIENT] Erreur lecture bloc ${chunk.id}:`, e);
                playNext();
            };
            audio.play().catch(e => {
                console.warn("[ARIA-CLIENT] Lecture bloquée par le navigateur:", e);
                playNext();
            });
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
                        console.log(`[ARIA-CLIENT] Réception de ${chunks.length} bloc(s) audio.`);
                        chunks.forEach(chunk => {
                            if (!window.playedIds.has(chunk.id)) {
                                window.playedIds.add(chunk.id);
                                window.audioQueue.push(chunk);
                            }
                        });
                        if (!window.isPlaying) playNext();
                    } catch(e) {
                        console.error("[ARIA-CLIENT] Erreur parsing bridge:", e);
                    }
                }
            });
        }

        const myVAD = await vad.MicVAD.new({
            modelURL: "/file=static/silero_v6.2/silero_vad.onnx",
            onSpeechStart: () => {
                console.log("[ARIA-VAD] Parole détectée...");
                const logo = document.querySelector('#aria_logo');
                if (logo) logo.classList.add('speaking');
            },
            onSpeechEnd: (audio) => {
                console.log("[ARIA-VAD] Fin de parole.");
                const logo = document.querySelector('#aria_logo');
                if (logo) logo.classList.remove('speaking');
                
                const wavBuffer = vad.utils.encodeWAV(audio);
                const base64Audio = "data:audio/wav;base64," + vad.utils.arrayBufferToBase64(wavBuffer);
                
                const container = document.getElementById('audio_input_box');
                const textarea = container ? container.querySelector('textarea') : null;
                
                if (textarea) {
                    textarea.value = base64Audio;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    setTimeout(() => {
                        const btn = document.querySelector('#aria_trigger');
                        if (btn) btn.click();
                    }, 50);
                }
            },
            positiveSpeechThreshold: 0.8,
            negativeSpeechThreshold: 0.4,
            minSpeechFrames: 3,
            redemptionFrames: 20,
        });
        
        myVAD.start();
        console.log("[ARIA-CLIENT] Système prêt (VAD Actif).");
    } catch (err) { console.error("[ARIA-ERROR] Initialisation échouée:", err); }
}
"""

# --- LOGIQUE SERVEUR ---
async def process_streaming_binaire(b64_audio, history):
    start_total = datetime.now()
    if not b64_audio: 
        yield history or [], ""
        return
    
    if history is None: history = []
    
    try:

        start_total = datetime.now()
        utils.log_info("app", "--- DEBUT Traitement (Stream Binaire)")
        # 1. Reception et STT
        header, encoded = b64_audio.split(",", 1)
        audio_data = base64.b64decode(encoded)
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_data)).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
        
        text_user = stt.transcribe_translate(samples)
        utils.log_info("app", f"   STT: reçu: '{text_user}' (Taille: {len(audio_data)} octets)")
        
        history.append({"role": "user", "content": text_user})
        history.append({"role": "assistant", "content": "..."})
        yield history, ""
        
        # 2. Appel LLM + TTS Streaming
        utils.log_info("app", f"   LLM: Début génération")
        response_gen = llm.get_answer_web(tts, text_user, "DefaultUser")
        chunk_count = 0
        stream_payload = []
        
        async for text_update, audio_chunk_path,textCleanedForAudio in response_gen:
            history[-1]["content"] = text_update
            
            payload_str = ""
            if audio_chunk_path and os.path.exists(audio_chunk_path):
                chunk_count += 1
                unique_id = f"bloc_{time.time_ns()}_{chunk_count}"
                
                # Lecture et encodage du son
                with open(audio_chunk_path, "rb") as f:
                    raw_son = f.read()
                    b64_data = f"data:audio/wav;base64,{base64.encodebytes(raw_son).decode('utf-8')}"
                
                # Debug : Sauvegarde si activé
                if DEBUG_SAVE_WAV:
                    save_path = os.path.join(TMP_DIR, f"{unique_id}.wav")
                    shutil.copy(audio_chunk_path, save_path)
                    utils.log_info("app", f"   TTS: > Bloc son généré: {unique_id} ({len(raw_son)} bytes) -> {unique_id}.wav, text: '{textCleanedForAudio}'")
                else:
                    utils.log_info("app", f"   TTS: > Bloc son généré: {unique_id} ({len(raw_son)} bytes) text: '{textCleanedForAudio}'")
                
                stream_payload.append({"id": unique_id, "data": b64_data})
                payload_str = json.dumps(stream_payload)
                
                # Nettoyage immédiat du fichier temporaire original
                try: os.remove(audio_chunk_path)
                except: pass
            
            yield history, payload_str
        
        utils.log_info("app", f"   LLM: FIN Traitement ({chunk_count} blocs son, Total: {(datetime.now() - start_total).total_seconds():.3f}s)")
        utils.log_info("app", f"   LLM: Réponse finale: '{history[-1]['content']}'")
                
        utils.log_info("app", f"--- FIN Traitement ({chunk_count} blocs, Total: {(datetime.now() - start_total).total_seconds():.3f}s)")
    except Exception as e:
        utils.log_info("app", f"!!! ERREUR CRITIQUE : {str(e)}")
        yield history, ""

# --- UI GRADIO ---
CSS = """
#audio_input_box, #audio_url_bridge, #aria_trigger {
    position: absolute;
    top: -9999px;
    left: -9999px;
    height: 0px !important;
    width: 0px !important;
    overflow: hidden;
    opacity: 0;
}

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
    
    # Bridge technique (visibles pour le DOM mais cachés par CSS)
    audio_url_bridge = gr.Textbox(elem_id="audio_url_bridge", visible=True)
    audio_input = gr.Textbox(elem_id="audio_input_box", visible=True)
    trigger_btn = gr.Button("Trigger", elem_id="aria_trigger", visible=True)
    
    with gr.Row():
        start_btn = gr.Button("🚀 ACTIVER MICRO & SON", variant="primary")
        
    start_btn.click(None, None, None, js=JS_COMBO)
    
    trigger_btn.click(
        fn=process_streaming_binaire, 
        inputs=[audio_input, chatbot], 
        outputs=[chatbot, audio_url_bridge],
        show_progress="hidden"
    )

if __name__ == "__main__":
    ariaHmi.launch(server_name="0.0.0.0", server_port=7860)