import gradio as gr
import numpy as np
import base64, io, json, os
from datetime import datetime
from pydub import AudioSegment
from components.stt import Stt
from components.llm import Llm
from components.tts import Tts
import components.utils as utils

# --- CONFIG CACHE ET MODÈLES ---
os.environ["HF_HOME"] = "/aria/models/huggingface"
os.environ["KOKORO_CACHE"] = "/aria/models/kokoro"
os.makedirs("/aria/models/huggingface", exist_ok=True)
os.makedirs("/aria/models/kokoro", exist_ok=True)

import warnings
import logging
import os

# 1. Supprime les warnings Python (UserWarning, FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Supprime les logs de chargement de Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Réduit le niveau de log des bibliothèques bruyantes
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# --- INIT COMPOSANTS ---
utils.log_perf("app", "--- DEBUT Chargement de la configuration et des modeles...")
with open("configs/default.json", "r") as f:
    config = json.load(f)

stt = Stt(config["Stt"]["params"])
llm = Llm(config["Llm"]["params"])
tts = Tts(config["Tts"]["params"])

# Correction de la KeyError : Initialisation de l'utilisateur par défaut
if "DefaultUser" not in llm.user_aware_messages:
    llm.user_aware_messages["DefaultUser"] = []

utils.log_perf("app", "--- FIN Chargement de la configuration et des modeles.")

def process_audio(b64_audio, history):
    if not b64_audio: return history, None
    if history is None: history = []
    
    start_step = datetime.now()
    start_total =     start_step
    
    utils.log_perf("app", "--- DEBUT Traitement nouvelle entrée audio")
    
    try:
        header, encoded = b64_audio.split(",", 1)
        audio_data = base64.b64decode(encoded)
        audio = AudioSegment.from_file(io.BytesIO(audio_data)).set_frame_rate(16000).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        utils.log_perf("app", f"Decodage du flux audio reçu (de {len(samples)/16000:.2f}s), en {((datetime.now() - start_step).total_seconds()):.3f}s")
        start_step = datetime.now()
    except Exception as e:
        utils.log_perf("app", f"Echec décodage: {e}")
        return history, None

    # 2. Transcription STT
    text_user = stt.transcribe_translate(samples)
    utils.log_perf("app", f"Transcription du flux effectuée en {((datetime.now() - start_step).total_seconds()):.3f}s")
    start_step = datetime.now()
    utils.log_perf("app", f"   Texte User : '{text_user}'")
    
    # FORMAT TUPLE : Compatible avec votre version de Gradio [[user, bot]]
    history.append([text_user, ""]) 
    yield history, None
    
    # 3. Réponse LLM & TTS
    response_gen = llm.get_answer_web(tts, text_user, "DefaultUser")
    utils.log_perf("app", f"Calcul réponse LMM effectuée en {((datetime.now() - start_step).total_seconds()):.3f}s")
    start_step = datetime.now()

    for text_update, audio_chunk in response_gen:
        # Mise à jour de l'élément assistant du dernier duo
        history[-1][1] = text_update 
        yield history, audio_chunk
    utils.log_perf("app", f"Envoi réponse au HMI effectué en {((datetime.now() - start_step).total_seconds()):.3f}s")
    start_step = datetime.now()

    utils.log_perf("app", f"--- FIN Traitement nouvelle entrée audio.(Terminé en {(datetime.now() - start_total).total_seconds():.3f}s)")

# --- JAVASCRIPT AVEC VAD INLINE ---
VAD_JS_CODE = """
async () => {
    try {
        console.log("Initialisation du contexte Audio...");
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        const processorCode = `
            class VADProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.isSpeaking = false;
                    this.silenceFrames = 0;
                    this.threshold = 0.05; 
                    this.silenceGracePeriod = 50; 
                }
                process(inputs) {
                    const input = inputs[0][0];
                    if (!input) return true;
                    const volume = Math.max(...input.map(Math.abs));
                    if (volume > this.threshold) {
                        if (!this.isSpeaking) {
                            this.isSpeaking = true;
                            this.port.postMessage('START');
                        }
                        this.silenceFrames = 0;
                    } else if (this.isSpeaking) {
                        this.silenceFrames++;
                        if (this.silenceFrames > this.silenceGracePeriod) {
                            this.isSpeaking = false;
                            this.port.postMessage('STOP');
                        }
                    }
                    return true;
                }
            }
            registerProcessor('vad-processor', VADProcessor);
        `;

        const blobModule = new Blob([processorCode], { type: 'application/javascript' });
        const moduleUrl = URL.createObjectURL(blobModule);
        await audioContext.audioWorklet.addModule(moduleUrl);
        
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioContext.createMediaStreamSource(stream);
        const vadNode = new AudioWorkletNode(audioContext, 'vad-processor');
        
        let mediaRecorder;
        let chunks = [];

        const setupRecorder = () => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
            mediaRecorder.onstop = async () => {
                if (chunks.length === 0) return;
                const blob = new Blob(chunks, { type: 'audio/wav' });
                // Correction erreur 187 : On ignore les fichiers trop petits (bruit/micro coupé)
                if (blob.size < 2000) { 
                    chunks = []; 
                    return; 
                }
                
                const reader = new FileReader();
                reader.onloadend = () => {
                    const container = document.getElementById('audio_input_box');
                    const input = container ? container.querySelector('textarea') : null;
                    const btn = document.getElementById('aria_trigger');
                    if (input && btn) {
                        input.value = reader.result;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        setTimeout(() => { btn.click(); }, 150);
                    }
                };
                reader.readAsDataURL(blob);
                chunks = [];
            };
        };

        vadNode.port.onmessage = (e) => {
            if (e.data === 'START') {
                setupRecorder();
                mediaRecorder.start();
                console.log("Aria écoute...");
            } else if (e.data === 'STOP') {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    console.log("Aria traite l'audio...");
                }
            }
        };

        source.connect(vadNode);
        alert("Aria est prête !");
    } catch (err) {
        console.error("ERREUR VAD : ", err);
        alert("Erreur micro : " + err.message);
    }
}
"""

with gr.Blocks(title="Aria Web") as ariaHmi:
    gr.Markdown("# 🎙️ Aria Web Interface")
    
    # On définit explicitement les types pour éviter l'auto-détection buggée
    chatbot = gr.Chatbot()
    audio_output = gr.Audio(interactive=False, autoplay=True, visible=True)
    
    start_btn = gr.Button("🚀 Activer le Micro")
    
    # Textbox pour l'injection audio
    audio_input = gr.Textbox(visible=False, elem_id="audio_input_box")
    trigger_btn = gr.Button("Trigger", visible=False, elem_id="aria_trigger")
    
    start_btn.click(None, None, None, js=VAD_JS_CODE)
    trigger_btn.click(process_audio, [audio_input, chatbot], [chatbot, audio_output])

if __name__ == "__main__":
    # LES 3 PARAMÈTRES CRUCIAUX :
    ariaHmi.launch(
        server_name="0.0.0.0", 
        server_port=7860
        # show_api=False,   # Désactive la génération du JSON d'API (cause du crash)
        # share=False,      # Évite de créer un tunnel SSH qui pourrait re-déclencher l'erreur
        # quiet=False
    )