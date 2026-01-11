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

# --- JAVASCRIPT AVEC LOGS CONSOLE WEB ET AUTO-SCROLL ---
JS_COMBO = """
async () => {
    if (window.ariaInitialized) return;
    window.ariaInitialized = true;
    console.log("[HMI] Système Aria Initialisé");
    
    window.audioQueue = [];
    window.playedIds = new Set();
    window.isPlaying = false;

    async function playNext() {
        if (window.audioQueue.length === 0) { window.isPlaying = false; return; }
        window.isPlaying = true;
        const chunk = window.audioQueue.shift();
        
        // LOG CONSOLE WEB : Lecture en cours
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
                            // LOG CONSOLE WEB : Identification du nouveau bloc reçu
                            console.log("[STREAM] Bloc reçu dans le navigateur : " + chunk.id);
                            
                            window.playedIds.add(chunk.id);
                            window.audioQueue.push(chunk);
                        }
                    });

                    // AUTO-SCROLL
                    setTimeout(() => {
                        const chatbotContainer = document.querySelector('#aria_chatbot .wrapper .bubble-wrap');
                        if (chatbotContainer) {
                            chatbotContainer.scrollTo({
                                top: chatbotContainer.scrollHeight,
                                behavior: 'smooth'
                            });
                        }
                    }, 150);

                    if (!window.isPlaying) playNext();
                } catch(e) {}
            }
        });
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new AudioContext({ sampleRate: 16000 });
        const code = `
            class VAD extends AudioWorkletProcessor {
                constructor() { super(); this.speaking = false; this.frames = 0; }
                process(inputs) {
                    const input = inputs[0][0];
                    if (!input) return true;
                    const vol = Math.max(...input.map(Math.abs));
                    if (vol > 0.1) {
                        if (!this.speaking) { this.speaking = true; this.port.postMessage('START'); }
                        this.frames = 0;
                    } else if (this.speaking) {
                        if (++this.frames > 50) { this.speaking = false; this.port.postMessage('STOP'); }
                    }
                    return true;
                }
            }
            registerProcessor('vad-processor', VAD);
        `;
        const blob = new Blob([code], { type: 'application/javascript' });
        await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));
        const source = audioContext.createMediaStreamSource(stream);
        const vadNode = new AudioWorkletNode(audioContext, 'vad-processor');
        let mr; let chunks = [];
        vadNode.port.onmessage = (e) => {
            if (e.data === 'START') { chunks = []; mr = new MediaRecorder(stream); mr.start(); }
            else if (e.data === 'STOP' && mr && mr.state === "recording") {
                mr.ondataavailable = (ev) => chunks.push(ev.data);
                mr.onstop = () => {
                    const blob = new Blob(chunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const input = document.querySelector('#audio_input_box textarea');
                        const btn = document.getElementById('aria_trigger');
                        if (input && btn) { 
                            input.value = reader.result; 
                            input.dispatchEvent(new Event('input', { bubbles: true })); 
                            btn.click(); 
                        }
                    };
                    reader.readAsDataURL(blob);
                };
                mr.stop();
            }
        };
        source.connect(vadNode);
    } catch (err) { console.error(err); }
}
"""

# --- LOGIQUE SERVEUR ---
def process_streaming_binaire(b64_audio, history):
    if not b64_audio: yield history, ""; return
    if history is None: history = []
    
    start_total = datetime.now()
    utils.log_perf("app", "--- DEBUT Traitement (Stream Binaire)")
    
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
                log_mode = "Disk" if DEBUG_SAVE_WAV else "Memory"
                
                with open(audio_chunk_path, "rb") as f:
                    bin_data = f.read()
                    b64_data = f"data:audio/wav;base64,{base64.b64encode(bin_data).decode('utf-8')}"
                
                # LOG CONSOLE SERVEUR
                utils.log_perf("app", f"   [STREAM] Bloc #{chunk_count} envoyé (ID: {unique_id}) - {log_mode} | User: \"{text_user}\"")
                
                if not DEBUG_SAVE_WAV:
                    try: os.remove(audio_chunk_path)
                    except: pass
                else:
                    shutil.copy(audio_chunk_path, os.path.join(TMP_DIR, f"{unique_id}.wav"))

                stream_payload.append({"id": unique_id, "data": b64_data})
                yield history, json.dumps(stream_payload)
            else:
                yield history, json.dumps(stream_payload)
        
        utils.log_perf("app", f"--- FIN Traitement ({chunk_count} blocs, Total: {(datetime.now() - start_total).total_seconds():.3f}s)")
                
    except Exception as e:
        utils.log_perf("app", f"!!! ERREUR : {str(e)}")
        yield history, ""

# --- UI ---
CSS = "#aria_chatbot .bubble-wrap { scroll-behavior: smooth; }"

with gr.Blocks(css=CSS, title="Aria Voice") as ariaHmi:
    gr.Markdown("# 🎙️ Aria Voice System")

    gr.Image(type="filepath", value="static/transition.gif",  height="100", elem_id="aria_logo")

    chatbot = gr.Chatbot(elem_id="aria_chatbot")
    audio_url_bridge = gr.Textbox(visible=False, elem_id="audio_url_bridge")
   
    start_btn = gr.Button("🚀 ACTIVER MICRO & SON")
    audio_input = gr.Textbox(visible=False, elem_id="audio_input_box")
    trigger_btn = gr.Button("Trigger", visible=False, elem_id="aria_trigger")
    
    start_btn.click(None, None, None, js=JS_COMBO)
    trigger_btn.click(
        fn=process_streaming_binaire, 
        inputs=[audio_input, chatbot], 
        outputs=[chatbot, audio_url_bridge], 
        show_progress="hidden"
    )

if __name__ == "__main__":
    ariaHmi.launch(server_name="0.0.0.0", server_port=7860)