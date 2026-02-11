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
utils.log_info("app", "--- DEBUT Chargement de la configuration...")
with open(_CONFIG_FILE, "r") as f:
    config = json.load(f)

# Extraction des modules actifs depuis la nouvelle section "active_modules"
active_vad = config["active_modules"]["vad"]
active_stt = config["active_modules"]["stt"]
active_llm = config["active_modules"]["llm"]
active_tts = config["active_modules"]["tts"]

# Instanciation dynamique basée sur le fichier de config
vad = Vad(config["vad"][active_vad]["params"])
stt = Stt(config["stt"][active_stt]["params"])
llm = Llm(config["llm"][active_llm]["params"], all_config=config)
tts = Tts(config["tts"][active_tts]["params"])
utils.log_info("app", f"--- FIN Chargement (VAD:{active_vad}, STT:{active_stt}, LLM:{active_llm}, TTS:{active_tts})")


def save_config(new_config_str):
    """Sauvegarde la nouvelle configuration JSON et recharge les composants si nécessaire."""
    try:
        new_data = json.loads(new_config_str)
        with open(_CONFIG_FILE, "w") as f:
            json.dump(new_data, f, indent=2)
        return "✅ Configuration sauvegardée avec succès ! (Redémarrez pour appliquer)"
    except Exception as e:
        return f"❌ Erreur lors de la sauvegarde : {str(e)}"

# --- LOGIQUE DE L'ONGLET PARAMÈTRES ---

def get_current_params(category, module_name):
    """Récupère dynamiquement le JSON des paramètres d'un module choisi"""
    try:
        # On extrait les paramètres de la catégorie (ex: 'llm') et du module (ex: 'Llm_Llama_local')
        params = config.get(category, {}).get(module_name, {}).get("params", {})
        return json.dumps(params, indent=2)
    except Exception as e:
        return "{}"

def update_config_ui(vad_choix, vad_json, stt_choix, stt_json, llm_choix, llm_json, tts_choix, tts_json):
    """Sauvegarde globale de la configuration modulaire"""
    try:
        # 1. Mise à jour des pointeurs de modules actifs
        config["active_modules"] = {
            "vad": vad_choix,
            "stt": stt_choix,
            "llm": llm_choix,
            "tts": tts_choix
        }

        # 2. Injection des paramètres modifiés par l'utilisateur
        config["vad"][vad_choix]["params"] = json.loads(vad_json)
        config["stt"][stt_choix]["params"] = json.loads(stt_json)
        config["llm"][llm_choix]["params"] = json.loads(llm_json)
        config["tts"][tts_choix]["params"] = json.loads(tts_json)

        # 3. Écriture physique sur le disque
        with open(_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
            
        return "✅ Configuration sauvegardée ! Redémarrez le système pour appliquer les changements."
    except json.JSONDecodeError as e:
        return f"❌ Erreur de syntaxe JSON dans l'un des blocs : {str(e)}"
    except Exception as e:
        return f"❌ Erreur sauvegarde : {str(e)}"


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

    with gr.Tabs():

        # --- ONGLET PRINCIPAL (CHAT) ---
        with gr.TabItem("💬 Interface Chat"):
            
            chatbot = gr.Chatbot(elem_id="aria_chatbot")
            
            # Bridge technique (visibles pour le DOM mais cachés par CSS)
            audio_url_bridge = gr.Textbox(elem_id="audio_url_bridge", visible=True)
            audio_input = gr.Textbox(elem_id="audio_input_box", visible=True)
            trigger_btn = gr.Button("Trigger", elem_id="aria_trigger", visible=True)
            
            with gr.Row():
                start_btn = gr.Button("🚀 ACTIVER MICRO & SON", variant="primary")
                
            start_btn.click(None, None, None, js=JS_COMBO)
            
        # --- NOUVEL ONGLET DE GESTION DES CONFIGS ---
        # --- ONGLET DE GESTION DES CONFIGS ---
        with gr.TabItem("⚙️ Paramètres"):
            gr.Markdown("### 🛠️ Configuration Modulaire")
            
            # --- BLOC 1 : VAD (Détection de voix) ---
            with gr.Group():
                gr.Markdown("#### 🎤 Détection de Voix (VAD)")
                # Liste déroulante : on prend les clés disponibles dans config['vad']
                # Valeur par défaut : celle définie dans active_modules
                vad_dd = gr.Dropdown(
                    choices=list(config["vad"].keys()),
                    value=config["active_modules"]["vad"],
                    label="Moteur VAD actif"
                )
                # Zone de code : affiche les params du moteur sélectionné par défaut
                vad_code = gr.Code(
                    value=json.dumps(config["vad"][config["active_modules"]["vad"]]["params"], indent=2),
                    language="json",
                    label="Paramètres du VAD",
                    lines=5
                )
                
                # Événement : Quand on change le dropdown, on met à jour le code JSON
                vad_dd.change(
                    fn=lambda x: get_current_params("vad", x),
                    inputs=[vad_dd],
                    outputs=[vad_code]
                )

            # --- BLOC 2 : STT (Transcription) ---
            with gr.Group():
                gr.Markdown("#### 📝 Transcription (STT)")
                stt_dd = gr.Dropdown(
                    choices=list(config["stt"].keys()),
                    value=config["active_modules"]["stt"],
                    label="Moteur STT actif"
                )
                stt_code = gr.Code(
                    value=json.dumps(config["stt"][config["active_modules"]["stt"]]["params"], indent=2),
                    language="json",
                    label="Paramètres du STT",
                    lines=8
                )
                stt_dd.change(
                    fn=lambda x: get_current_params("stt", x),
                    inputs=[stt_dd],
                    outputs=[stt_code]
                )

            # --- BLOC 3 : LLM (Cerveau) ---
            with gr.Group():
                gr.Markdown("#### 🧠 Intelligence (LLM)")
                llm_dd = gr.Dropdown(
                    choices=list(config["llm"].keys()),
                    value=config["active_modules"]["llm"],
                    label="Modèle LLM actif"
                )
                llm_code = gr.Code(
                    value=json.dumps(config["llm"][config["active_modules"]["llm"]]["params"], indent=2),
                    language="json",
                    label="Paramètres du LLM (System prompt, URL...)",
                    lines=10
                )
                llm_dd.change(
                    fn=lambda x: get_current_params("llm", x),
                    inputs=[llm_dd],
                    outputs=[llm_code]
                )

            # --- BLOC 4 : TTS (Synthèse vocale) ---
            with gr.Group():
                gr.Markdown("#### 🗣️ Synthèse Vocale (TTS)")
                tts_dd = gr.Dropdown(
                    choices=list(config["tts"].keys()),
                    value=config["active_modules"]["tts"],
                    label="Moteur TTS actif"
                )
                tts_code = gr.Code(
                    value=json.dumps(config["tts"][config["active_modules"]["tts"]]["params"], indent=2),
                    language="json",
                    label="Paramètres du TTS",
                    lines=8
                )
                tts_dd.change(
                    fn=lambda x: get_current_params("tts", x),
                    inputs=[tts_dd],
                    outputs=[tts_code]
                )

            # --- BOUTON DE SAUVEGARDE GLOBALE ---
            gr.Markdown("---")
            save_btn = gr.Button("💾 SAUVEGARDER TOUTE LA CONFIGURATION", variant="primary")
            status_msg = gr.Markdown("")
            
            # Au clic, on envoie toutes les valeurs actuelles (dropdowns + codes)
            save_btn.click(
                fn=update_config_ui,
                inputs=[
                    vad_dd, vad_code,
                    stt_dd, stt_code,
                    llm_dd, llm_code,
                    tts_dd, tts_code
                ],
                outputs=[status_msg]
            )


    trigger_btn.click(
        fn=process_streaming_binaire, 
        inputs=[audio_input, chatbot], 
        outputs=[chatbot, audio_url_bridge],
        show_progress="hidden"
    )

if __name__ == "__main__":
    ariaHmi.launch(server_name="0.0.0.0", server_port=7860)