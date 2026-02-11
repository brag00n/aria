from datetime import datetime
import os
import sys
import pathlib

def checkpoint(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")
    
checkpoint("1. Avant OS ENV")

# --- VARIABLES D'ENVIRONNEMENT (CRITIQUE) ---
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ["HF_HOME"] = "models/huggingface"
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'

# --- ACTIVATION DU CACHE CUDA (SPECIFIQUE LINUX/DOCKER) ---
if os.name != 'nt':  # On ne le fait PAS si c'est Windows ('nt')
    cuda_cache_dir = "/aria/.cuda_cache"
    try:
        pathlib.Path(cuda_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ['CUDA_CACHE_PATH'] = cuda_cache_dir
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648' 
        checkpoint("1.5. Cache CUDA configuré (Linux Only)")
    except Exception as e:
        print(f"Erreur cache CUDA: {e}")
else:
    checkpoint("1.5. Utilisation du cache CUDA natif Windows")

checkpoint("2. Avant Imports torch")
import torch
checkpoint("3. Torch chargé")
import transformers
checkpoint("4. Transformers chargé")
import gradio as gr
checkpoint("5. Gradio chargé")

import warnings
import logging

#from prototypes.test_llm_mcp.test_llm_mcp import CONFIG_FILE

# --- SILENCE AUX WARNINGS (AVANT TOUT IMPORT) ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["LLAMA_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*symlinks.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

#import gradio as gr
checkpoint("6. Avant imports numpy")
import numpy as np
checkpoint("6. numpy chargé")
import base64, io, json, shutil, time, re
checkpoint("6. base64, io, json, shutil, time, re chargé")
from pydub import AudioSegment
checkpoint("6. AudioSegment chargé")
from components.stt import Stt
from components.llm import Llm
from components.tts import Tts
from components.vad import Vad
checkpoint("6. Stt, Llm, Tts chargés")

import components.utils as utils
checkpoint("6. utils chargés")

# --- CONFIGURATION ---
DEBUG_SAVE_WAV = False
_TMP_DIR = "/aria/tmp/tts"
_CONFIG_FILE = "configs/default.json"

if os.path.exists(_TMP_DIR):
    shutil.rmtree(_TMP_DIR)
os.makedirs(_TMP_DIR, mode=0o777, exist_ok=True)

# --- INIT COMPOSANTS ---
utils.log_info("app", "--- DEBUT Chargement de la configuration et des modeles...")
with open(_CONFIG_FILE, "r") as f:
    config = json.load(f)

vad = Vad(config["Vad_Web"]["params"])
stt = Stt(config["Stt_FasterWhisper"]["params"])
llm = Llm(config["Llm_Ministral_lmstudio"]["params"], all_config=config)
tts = Tts(config["Tts_Sherpa"]["params"])
utils.log_info("app", "--- FIN Chargement de la configuration et des modeles.")

JS_COMBO = f"""
async () => {{
    // 1. Injection des fonctions du VAD
    {vad.get_javascript()}

    // 2. Initialisation du streaming audio TTS
    {tts.get_javascript()}
}}
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
                    save_path = os.path.join(_TMP_DIR, f"{unique_id}.wav")
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