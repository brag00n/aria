from datetime import datetime
import os
import sys
import pathlib

import components.utils as utils
utils.log_info("app", "--- DEBUT Chargement des librairies...")

# --- VARIABLES D'ENVIRONNEMENT (CRITIQUE) ---
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ["HF_HOME"] = "models/huggingface"
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'

# --- ACTIVATION DU CACHE CUDA (SPECIFIQUE LINUX/DOCKER) ---
if os.name != 'nt':  # On ne le fait PAS si c'est Windows ('nt')
    cuda_cache_dir = utils.aria_path(".cuda_cache")
    try:
        pathlib.Path(cuda_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ['CUDA_CACHE_PATH'] = cuda_cache_dir
        os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648' 
        utils.log_info("app","Cache CUDA configuré (Linux Only)")
    except Exception as e:
        print(f"Erreur cache CUDA: {e}")
else:
    utils.log_info("app","Utilisation du cache CUDA natif Windows")

import torch
utils.log_info("app","Torch chargé")
import transformers
utils.log_info("app","Transformers chargé")
import gradio as gr
utils.log_info("app","Gradio chargé")

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
import numpy as np
utils.log_info("app","numpy chargé")
import base64, io, json, shutil, time, re, threading
utils.log_info("app","base64, io, json, shutil, time, re chargé")
from pydub import AudioSegment
utils.log_info("app","AudioSegment chargé")
from components.stt import Stt
from components.llm import Llm
from components.tts import Tts
from components.vad import Vad
from components import visualizer, mic_toggle
from components.language import Language, normalize
from components.conversation import Conversation
from components import https
from components import memory_tab, agents_tab, audit_retention, echanges, settings as settings_mod
from components import interlocuteur
from components import history as history_mod
utils.log_info("app","Stt, Llm, Tts chargés")


# --- CONFIGURATION ---
DEBUG_SAVE_WAV = False
_TMP_DIR = utils.aria_path("tmp", "tts")
_CONFIG_FILE = "configs/default.json"
from components.agents import bridge as agents_bridge
chat_aria = None                                    # chat de l'interface via l'agent par défaut `aria` (components/agents/chat.py), créé au démarrage des agents
_boucle_interface = agents_bridge.LoopHolder()      # boucle de l'interface (Gradio), lue par le pont de l'agent par défaut `aria`
_boucle_mcp = {}          # boucle asyncio qui porte les sessions MCP (posée au démarrage de l'application)
agents_srv = None          # serveur de l'API des agents (components/agents/server.py), None si la fonction est désactivée

if os.path.exists(_TMP_DIR):
    shutil.rmtree(_TMP_DIR)
os.makedirs(_TMP_DIR, mode=0o777, exist_ok=True)

utils.log_info("app", "--- FIN Chargement des librairies...")

# --- INIT COMPOSANTS ---
utils.log_info("app", "--- DEBUT Chargement de la configuration...")
with open(_CONFIG_FILE, "r") as f:
    config = json.load(f)

# Section "global" : une langue unique pour tous les modules (components/language.py).
# Migre aussi l'ancien "active_modules" placé à la racine de la config.
language = Language.from_config(config)
utils.log_info("LANG", f"Langue globale '{language.code}' (repli '{language.default}') | VAD : indépendant de la langue")

# Extraction des modules actifs depuis la section "active_modules"
active_vad = config["active_modules"]["vad"]
active_stt = config["active_modules"]["stt"]
active_llm = config["active_modules"]["llm"]
active_tts = config["active_modules"]["tts"]

# Instanciation dynamique basée sur le fichier de config
vad = Vad(config["vad"][active_vad]["params"])
stt = Stt(config["stt"][active_stt]["params"], language=language)
# Secret du canal HTTP de la mémoire : généré à CHAQUE démarrage, en mémoire vive seulement. Il passe par une copie de la config réservée au LLM
# d'Aria (jamais dans `config`, qui est réécrite dans default.json) ; le gestionnaire MCP le transmet au processus mémoire par son environnement.
from components.agents import compat as agents_compat, memory_link
agents_compat.migrate_config(config)          # ancienne section « access » -> « agents » (en mémoire ; le fichier est réécrit à la prochaine sauvegarde des Paramètres)
llm = Llm(config["llm"][active_llm]["params"], all_config={**config, "_memory_secret": memory_link.new_secret()}, language=language)
tts = Tts(config["tts"][active_tts]["params"], language=language)
conv = Conversation(config.get("conversation"))   # sessions de conversation + mots d'arrêt
conv_audit = echanges.ConversationAria()          # identifiant de la conversation courante de l'interface, pour le journal d'audit
_audit = config.get("audit") or {}                # journal d'audit des échanges (components/echanges.py) : actif par défaut, 90 jours, hors git
echanges.configure(utils.aria_path(*str(_audit.get("dir", "models/audit")).split("/")), enabled=_audit.get("enabled", True),
                   log_text=_audit.get("log_text", True), retention_days=_audit.get("retention_days", echanges.DEFAULT_RETENTION_DAYS))
utils.log_info("app", f"--- FIN Chargement (VAD:{active_vad}, STT:{active_stt}, LLM:{active_llm}, TTS:{active_tts})")

def read_logs_memory():
    """Appelle le buffer mémoire de utils"""
    return utils.get_logs_from_memory()

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

def update_config_ui(global_json, vad_choix, vad_json, conv_json, memory_json, stt_choix, stt_json, llm_choix, llm_json, tts_choix, tts_json, mcp_json,
                     profiles_json="", close_open=False, audit_json=""):
    """Sauvegarde globale de la configuration modulaire. Tout est analysé et validé AVANT la moindre modification de `config` (components/settings.py) :
    une erreur dans n'importe quel champ ne laisse ni la mémoire ni le fichier à moitié modifiés."""
    try:
        fields = {"global": global_json, "conversation": conv_json, "memory": memory_json, "mcp": mcp_json, "profiles": profiles_json, "audit": audit_json,
                  "vad": vad_json, "stt": stt_json, "llm": llm_json, "tts": tts_json,
                  "vad_choice": vad_choix, "stt_choice": stt_choix, "llm_choice": llm_choix, "tts_choice": tts_choix}
        parsed, changed = settings_mod.apply_settings(config, fields)
        settings_mod.write_config(config, _CONFIG_FILE)
        if parsed["audit"] is not None:                 # activer, durée et texte/empreinte valent tout de suite ; le dossier au prochain démarrage
            a = config["audit"]
            echanges.configure(echanges.directory(), enabled=a.get("enabled", True), log_text=a.get("log_text", True),
                               retention_days=a.get("retention_days", echanges.DEFAULT_RETENTION_DAYS))
        closed, warnings = 0, []
        if changed and agents_srv is not None:
            from components.agents import server as agents_server
            used = agents_server.keys_using_profiles(agents_srv, changed)
            if used:
                warnings.append("Des clés utilisent un profil modifié ou supprimé : " + ", ".join(used)
                                + ". Un profil supprimé donne une erreur à ces clés ; cochez « Fermer les conversations ouvertes » pour appliquer le changement tout de suite.")
            if close_open:
                closed = agents_server.close_sessions_for_profiles(agents_srv, changed)
        return settings_mod.save_message(changed, closed, warnings)
    except json.JSONDecodeError as e:
        return f"❌ Erreur de syntaxe JSON dans l'un des blocs : {str(e)}"
    except ValueError as e:
        return str(e) if str(e).startswith("❌") else f"❌ {e}"
    except Exception as e:
        return f"❌ Erreur sauvegarde : {str(e)}"


JS_COMBO = f"""
(async () => {{
    // 0. Visualiseur audio (isolé : une erreur ici n'empêche ni le VAD ni la lecture)
    {visualizer.build_js(config.get('visualizer'))}

    // 1. Injection des fonctions du VAD
    {vad.get_javascript()}

    // 2. Initialisation du streaming audio TTS
    {tts.get_javascript()}
}})()
"""

# JavaScript exécuté au chargement de la page : clic sur le cadre = micro et son activés / coupés
JS_ON_LOAD = "() => {" + mic_toggle.js_toggle(JS_COMBO, conv.enabled) + "}"

# --- LOGIQUE SERVEUR ---
async def process_streaming_binaire(b64_audio, history, interlocutor=""):
    start_total = datetime.now()
    if not b64_audio:
        yield history or [], ""
        return
    
    if history is None: history = []
    
    try:

        start_total = datetime.now()
        utils.log_info("app", "--- DEBUT Traitement (Stream Binaire)")
        timer = utils.StepTimer("app")
        # 1. Reception et STT
        header, encoded = b64_audio.split(",", 1)
        audio_data = base64.b64decode(encoded)
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_data)).set_frame_rate(16000).set_channels(1)
        # Normalisation selon la largeur réelle des échantillons (le navigateur envoie du 32 bits, pas du 16 bits)
        full_scale = float(1 << (8 * audio_seg.sample_width - 1))
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / full_scale
        audio_sec = len(samples) / 16000
        timer.mark("decodage_audio", f"audio de {audio_sec:.1f}s")
        
        text_user = stt.transcribe_translate(samples)
        utils.log_info("app", f"   STT: reçu: '{text_user}' (Taille: {len(audio_data)} octets)")
        stt_dt = time.perf_counter() - timer.last
        timer.mark("stt", f"{stt_dt / max(audio_sec, 0.01):.2f}x la durée de l'audio")
        if not re.search(r"\w", text_user or ""):
            # Rien d'exploitable (silence, bruit écarté par le filtre VAD) : on n'appelle pas le LLM,
            # la passerelle rejette un message vide (erreur 400)
            utils.log_info("app", "   STT: aucune parole détectée, enregistrement ignoré")
            yield history, ctl_payload("idle")
            return

        # Mots-clés : arrêt d'Aria, ouverture / clôture de conversation (components/conversation.py)
        action = conv.classify(text_user, from_voice=True)
        if action == "stop":
            conv.request_stop()
            yield history, ctl_payload("stop")
            return
        if action == "ignore":
            utils.log_info("CONV", f"Hors conversation, phrase ignorée : '{text_user}'")
            yield history, ctl_payload("idle")
            return
        if action == "open":
            conv.begin()
            conv_audit.renew()
            yield history, ctl_payload("conv_open")      # l'icône du cadre apparaît (chargement) pendant que la réponse se prépare

        async for update in repondre(text_user, history, timer, start_total, interlocutor, voix=True):
            yield update
        if action == "close":
            conv.end()
            conv_audit.renew()
            forget_context()                             # clôture : Aria oublie la conversation (réglage memory.reset_on_close)
            await reset_interlocutor()                   # ... et ne sait plus à qui elle parlait
            yield history, ctl_payload("conv_close")     # l'icône du cadre disparaît après la fin de la réponse
    except Exception as e:
        utils.log_info("app", f"!!! ERREUR CRITIQUE : {str(e)}")
        yield history, ""

def ctl_payload(kind):
    """Ordre de contrôle envoyé au navigateur par le pont audio : stop, idle, conv_open, conv_close."""
    return json.dumps([{"id": f"ctl_{kind}_{time.time_ns()}", "ctl": kind}])


STOP_NOTICE = "\n\n> ⏹️ Réponse interrompue\n\n"


async def repondre(text_user, history, timer, start_total, interlocutor="", voix=False):
    """Une seule réponse à la fois (verrou) ; l'arrêt, lui, ne passe jamais par le verrou."""
    async with conv.lock:
        conv.stop_requested = False
        async for update in _generer(text_user, history, timer, start_total, interlocutor, voix):
            yield update


async def _generer(text_user, history, timer, start_total, interlocutor="", voix=False):
    """Réponse d'Aria à un message de l'utilisateur (voix transcrite ou texte saisi) : texte en continu + blocs son."""
    history.append({"role": "user", "content": text_user})
    history.append({"role": "assistant", "content": "..."})
    yield history, ""

    # Qui parle ? La liste déroulante fait référence : on la communique au serveur de mémoire (qui garde l'état, même
    # après un redémarrage de ce dernier), et on récupère la consigne du tour. Sans serveur de mémoire : Inconnue, sans effet.
    etat = await interlocuteur.etat(mcp_manager(), interlocutor or "")
    # Journal d'audit : interlocuteur actif d'Aria (libellé de la liste, « Inconnue » sinon), conversation de l'interface, voix ou clavier
    qui = next((label for label, pid in etat.get("choices", []) if pid == etat.get("actif")), "") or "Inconnue"
    echanges.start("aria", qui, conv_audit.id, voix=bool(voix), query=text_user)

    # 2. Appel LLM + TTS Streaming
    utils.log_info("app", f"   LLM: Début génération")
    response_gen = llm.get_answer_web(tts, text_user, "DefaultUser", etat["contexte"])
    chunk_count = 0
    first_text = False
    
    async for text_update, audio_chunk_path,textCleanedForAudio in response_gen:
        if conv.stop_requested:
            # Arrêt demandé (bouton, mot-clé dit ou écrit) : on coupe la génération, le navigateur coupe la lecture
            await response_gen.aclose()
            history[-1]["content"] = (text_update or "") + STOP_NOTICE
            utils.log_info("CONV", "⏹️ Réponse interrompue")
            yield history, ctl_payload("stop")
            return
        history[-1]["content"] = text_update
        if text_update and not first_text:
            first_text = True
            timer.mark("premier_texte_llm")
        
        payload_str = ""
        if audio_chunk_path and os.path.exists(audio_chunk_path):
            chunk_count += 1
            timer.mark("premier_son_pret" if chunk_count == 1 else f"bloc_audio_{chunk_count}")
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
            
            # Un seul bloc par envoi : renvoyer la liste cumulée à chaque jeton saturait le navigateur
            # (le client ignore une valeur vide et filtre les identifiants déjà vus)
            payload_str = json.dumps([{"id": unique_id, "data": b64_data}])
            
            # Nettoyage immédiat du fichier temporaire original
            try: os.remove(audio_chunk_path)
            except: pass
        
        yield history, payload_str
    
    utils.log_info("app", f"   LLM: FIN Traitement ({chunk_count} blocs son, Total: {(datetime.now() - start_total).total_seconds():.3f}s)")
    utils.log_info("app", f"   LLM: Réponse finale: '{utils.answer_for_log(history[-1]['content'])}'")
    timer.mark("fin_traitement")
            
    utils.log_info("app", f"--- FIN Traitement ({chunk_count} blocs, Total: {(datetime.now() - start_total).total_seconds():.3f}s)")


async def process_texte(text, history, interlocutor=""):
    """Message saisi au clavier : même chaîne que la voix, sans la reconnaissance vocale.
    La réponse est lue à voix haute si le son est activé (clic sur le cadre), sinon elle reste écrite."""
    text = (text or "").strip()
    if not text:
        yield history or [], "", ""
        return
    if history is None: history = []
    try:
        action = conv.classify(text, from_voice=False)
        if action == "stop":
            conv.request_stop()
            yield history, ctl_payload("stop"), ""
            return
        utils.log_info("app", f"--- DEBUT Traitement (texte saisi) : '{text}'")
        timer = utils.StepTimer("app")
        async for hist, payload in repondre(text, history, timer, datetime.now(), interlocutor):
            yield hist, payload, ""
        if action == "close":
            conv.end()
            conv_audit.renew()
            forget_context()
            await reset_interlocutor()
            yield history, ctl_payload("conv_close"), ""
    except Exception as e:
        utils.log_info("app", f"!!! ERREUR CRITIQUE : {str(e)}")
        yield history, "", ""


def forget_context(force=False):
    """Efface l'historique envoyé au LLM (pas le chat affiché). Automatique à la clôture d'une conversation si
    memory.reset_on_close est vrai ; `force` pour le bouton « Nouvelle conversation »."""
    if not (force or history_mod.settings(config.get("memory"))["reset_on_close"]):
        return
    store = getattr(llm.engine, "user_aware_messages", None)
    if store is not None:
        history_mod.reset(store)


def mcp_manager():
    """Gestionnaire MCP du moteur LLM actif (None pour le moteur local, qui n'utilise pas ce module)."""
    return getattr(llm.engine, "mcp_manager", None)


async def reset_interlocutor():
    """Clôture / nouvelle conversation : Aria ne sait plus à qui elle parlait (retour à « Inconnue »)."""
    await interlocuteur.etat(mcp_manager(), "")


_last_interlocutor = {"id": ""}      # dernier interlocuteur connu de l'interface (choisi à la main OU fixé par Aria)


async def refresh_interlocutor():
    """Met à jour la liste déroulante à partir du serveur de mémoire (nouvelles personnes, interlocuteur actif)."""
    etat = await interlocuteur.etat(mcp_manager(), "?")
    _last_interlocutor["id"] = etat["actif"]          # suit aussi les changements décidés par Aria (outil identifier_interlocuteur)
    return gr.update(choices=etat["choices"], value=etat["actif"], visible=interlocuteur.disponible(mcp_manager()))


async def change_interlocutor(choice):
    """Changement à la main dans la liste : on prévient le serveur. Passer d'une personne connue à une autre efface
    l'historique du LLM, pour qu'il ne reste pas de confidences de la précédente dans la conversation."""
    choice = choice or ""
    previous, _last_interlocutor["id"] = _last_interlocutor["id"], choice
    await interlocuteur.etat(mcp_manager(), choice)
    if previous and choice and previous != choice:
        forget_context(force=True)
        conv_audit.renew()
        utils.log_info("INTERLOCUTEUR", f"Changement {previous} -> {choice} : historique du LLM effacé")


async def new_conversation():
    """Bouton « Nouvelle conversation » : Aria oublie tout, le chat est vidé, la conversation vocale se referme."""
    conv.request_stop()
    forget_context(force=True)
    conv_audit.renew()
    if conv.active:
        conv.end()
    await reset_interlocutor()
    return [], ctl_payload("conv_close")


def stop_aria():
    """Bouton « Stop » : coupe la génération en cours ; le navigateur coupe aussi la lecture (JS du bouton + ordre « stop »)."""
    conv.request_stop()
    return ctl_payload("stop")


# =====================================================================================================================================
# CHAT VIA L'AGENT PAR DÉFAUT « aria » (components/agents/chat.py) : même cœur de tour que l'API des agents, en interne (sans HTTP, sans clé).
# Actif quand `agents.chat_via_agent` est vrai ; sinon l'ancien chemin ci-dessus reste en service (retour arrière sans revert).
# =====================================================================================================================================
def chat_via_agent():
    """Le chat passe-t-il par l'agent `aria` ? (lu dans la config vivante : l'administrateur peut revenir en arrière sans toucher au code)"""
    return bool(agents_compat.section(config).get("chat_via_agent", False)) and chat_aria is not None


def _chat():
    return chat_aria


async def _agent_tour(history, texte=None, audio=None, interlocutor="", voix=False):
    """Un tour du chat via l'agent : met à jour l'historique d'affichage et produit (historique, charge utile du pont audio) comme `_generer`."""
    texte_affiche = texte
    entame = False
    async for nom, d in _chat().tour(texte=texte, audio=audio, interlocuteur_choisi=interlocutor, speak=True):
        if nom == "user_text":
            texte_affiche = d
            history.append({"role": "user", "content": d})
            history.append({"role": "assistant", "content": "..."})
            entame = True
            yield history, ""
        elif nom == "ctl":
            yield history, ctl_payload(d)
        elif nom == "texte":
            if not entame:
                history.append({"role": "user", "content": texte_affiche})
                history.append({"role": "assistant", "content": "..."})
                entame = True
            history[-1]["content"] = d
            yield history, ""
        elif nom == "audio":
            path = d["path"]
            if not (path and os.path.exists(path)):
                continue
            with open(path, "rb") as f:
                raw_son = f.read()
            unique_id = f"bloc_{time.time_ns()}"
            b64_data = f"data:audio/wav;base64,{base64.encodebytes(raw_son).decode('utf-8')}"
            utils.log_info("app", f"   TTS: > Bloc son généré: {unique_id} ({len(raw_son)} bytes) text: '{d.get('text', '')}'")
            try: os.remove(path)
            except OSError: pass
            yield history, json.dumps([{"id": unique_id, "data": b64_data}])
        elif nom == "erreur":
            if not entame:
                history.append({"role": "user", "content": texte_affiche})
                history.append({"role": "assistant", "content": ""})
                entame = True
            history[-1]["content"] = f"Erreur : {d}"
            yield history, ""
    if entame and history and history[-1].get("content") == "...":
        history[-1]["content"] = ""                                    # tour sans réponse (arrêt immédiat) : pas de « ... » qui reste affiché


async def agent_process_streaming_binaire(b64_audio, history, interlocutor=""):
    if not b64_audio:
        yield history or [], ""
        return
    if history is None:
        history = []
    try:
        async for update in _agent_tour(history, audio=b64_audio, interlocutor=interlocutor, voix=True):
            yield update
    except Exception as e:
        utils.log_info("app", f"!!! ERREUR CRITIQUE : {str(e)}")
        yield history, ""


async def agent_process_texte(text, history, interlocutor=""):
    text = (text or "").strip()
    if not text:
        yield history or [], "", ""
        return
    if history is None:
        history = []
    try:
        async for hist, payload in _agent_tour(history, texte=text, interlocutor=interlocutor):
            yield hist, payload, ""
    except Exception as e:
        utils.log_info("app", f"!!! ERREUR CRITIQUE : {str(e)}")
        yield history, "", ""


async def agent_new_conversation():
    await _chat().nouvelle_conversation()
    return [], ctl_payload("conv_close")


def agent_stop():
    _chat().arreter()
    return ctl_payload("stop")


async def agent_refresh_interlocutor():
    etat, dispo = await _chat().rafraichir_interlocuteur()
    return gr.update(choices=etat["choices"], value=etat["actif"], visible=dispo)


async def agent_change_interlocutor(choice):
    await _chat().changer_interlocuteur(choice)


# --- Aiguillage : chaque évènement de l'interface appelle l'une de ces fonctions, qui choisit le chemin À CHAQUE appel ---
async def route_process_streaming_binaire(b64_audio, history, interlocutor=""):
    gen = agent_process_streaming_binaire(b64_audio, history, interlocutor) if chat_via_agent() else process_streaming_binaire(b64_audio, history, interlocutor)
    async for update in gen:
        yield update


async def route_process_texte(text, history, interlocutor=""):
    gen = agent_process_texte(text, history, interlocutor) if chat_via_agent() else process_texte(text, history, interlocutor)
    async for update in gen:
        yield update


async def route_new_conversation():
    return await (agent_new_conversation() if chat_via_agent() else new_conversation())


def route_stop_aria():
    return agent_stop() if chat_via_agent() else stop_aria()


async def route_refresh_interlocutor():
    return await (agent_refresh_interlocutor() if chat_via_agent() else refresh_interlocutor())


async def route_change_interlocutor(choice):
    return await (agent_change_interlocutor(choice) if chat_via_agent() else change_interlocutor(choice))


def request_restart(confirmed=""):
    """Demande de redémarrage depuis l'interface : tracée dans les logs, puis le serveur se termine dans 2 s.
    Le conteneur est relancé par sa politique de redémarrage (docker-compose : restart: unless-stopped).
    `confirmed` vient de la boîte de dialogue du navigateur : « Annuler » ne redémarre rien."""
    if confirmed != "oui":
        utils.log_info("app", f"Redémarrage annulé depuis l'interface (valeur reçue du navigateur : {confirmed!r})")
        return "Redémarrage annulé."
    utils.log_info("app", "🔄 REDÉMARRAGE DEMANDÉ depuis l'interface (onglet Paramètres)")
    def _exit_later():
        time.sleep(2)           # laisse le temps de répondre au navigateur et d'écrire le log
        utils.log_info("app", "🔄 Arrêt du serveur Aria : le conteneur va le relancer")
        os._exit(0)             # arrêt immédiat, sans attendre les threads (modèles chargés, serveur Gradio)
    threading.Thread(target=_exit_later, daemon=True).start()
    return "🔄 Redémarrage en cours... rechargez la page dans une vingtaine de secondes."

# JavaScript du bouton « Redémarrer » : confirmation, puis une boîte « Redémarrage en cours... » animée qui teste la connexion.
# Juste après le clic le serveur répond encore (~2 s) : on attend donc d'abord de le voir TOMBER, puis REVENIR avant de conclure.
# Au retour la page est rechargée : la session Gradio ouverte avant l'arrêt n'existe plus côté serveur.
JS_RESTART = r"""() => {
    if (!confirm("Redémarrer le serveur Aria ? Les conversations en cours seront coupées.")) return ["non"];
    if (document.getElementById("aria_restart_box")) return ["oui"];
    const box = document.createElement("div");
    box.id = "aria_restart_box";
    const card = document.createElement("div");
    card.className = "aria-restart-card";
    const spinner = document.createElement("div");
    spinner.className = "aria-restart-spinner";
    const text = document.createElement("div");
    text.className = "aria-restart-text";
    text.textContent = "Redémarrage en cours...";
    const sub = document.createElement("div");
    sub.className = "aria-restart-sub";
    sub.textContent = "Test de la connexion au serveur";
    card.append(spinner, text, sub);
    box.append(card);
    document.body.append(box);

    const started = Date.now();
    let sawDown = false;
    const alive = async () => {
        const ctl = new AbortController();
        const timer = setTimeout(() => ctl.abort(), 2000);
        try {
            const r = await fetch("/?_=" + Date.now(), { cache: "no-store", signal: ctl.signal });
            return r.ok;
        } catch (e) { return false; }
        finally { clearTimeout(timer); }
    };
    const fail = (msg) => {
        spinner.style.display = "none";
        text.textContent = msg;
        sub.textContent = "";
        const close = document.createElement("button");
        close.textContent = "Fermer";
        close.onclick = () => box.remove();
        card.append(close);
    };
    const tick = async () => {
        const up = await alive();
        const sec = (Date.now() - started) / 1000;
        if (!up) sawDown = true;
        if (up && sawDown) {
            spinner.style.display = "none";
            text.textContent = "Aria est de retour";
            sub.textContent = "Rechargement de la page...";
            setTimeout(() => location.reload(), 700);
            return;
        }
        if (up && sec > 25) return fail("Le serveur n'a pas redémarré (il répond toujours).");
        if (sec > 300) return fail("Aria ne répond pas après 5 minutes. Consultez les logs du conteneur.");
        sub.textContent = sawDown ? "Serveur arrêté, attente de son retour (" + Math.round(sec) + " s)" : "Arrêt du serveur en cours...";
        setTimeout(tick, 1500);
    };
    setTimeout(tick, 1500);
    return ["oui"];
}"""

# --- UI GRADIO ---
CSS = """
/* Boîte « Redémarrage en cours » (JS_RESTART) */
#aria_restart_box { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center;
    background: rgba(0, 0, 0, 0.55); }
#aria_restart_box .aria-restart-card { background: var(--background-fill-primary, #fff); color: var(--body-text-color, #111);
    border-radius: 12px; padding: 28px 36px; text-align: center; min-width: 280px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35); }
#aria_restart_box .aria-restart-spinner { width: 42px; height: 42px; margin: 0 auto 14px; border-radius: 50%;
    border: 4px solid rgba(128, 128, 128, 0.3); border-top-color: #e08a00; animation: aria-spin 0.9s linear infinite; }
#aria_restart_box .aria-restart-text { font-size: 1.15em; font-weight: 600; }
#aria_restart_box .aria-restart-sub { margin-top: 6px; opacity: 0.7; font-size: 0.9em; }
#aria_restart_box button { margin-top: 14px; padding: 6px 16px; cursor: pointer; }
@keyframes aria-spin { to { transform: rotate(360deg); } }

#audio_input_box, #audio_url_bridge, #aria_trigger, #restart_confirm_box {
    position: absolute;
    top: -9999px;
    left: -9999px;
    height: 0px !important;
    width: 0px !important;
    overflow: hidden;
    opacity: 0;
}


/* Ligne d'information système dans le chat (ex. « Outil utilisé ») : couleur distincte, jamais lue à voix haute */
#aria_chatbot blockquote {
    color: #d97706 !important;
    border-left: 3px solid #d97706 !important;
    background: rgba(217, 119, 6, 0.08) !important;
    margin: 4px 0 !important;
    padding: 2px 10px !important;
    font-size: 0.9em;
}
#aria_chatbot blockquote p, #aria_chatbot blockquote code { color: inherit !important; }

#log_display textarea {
    font-family: 'Courier New', monospace !important;
    background-color: #0d1117 !important;
    color: #33ff33 !important;
    white-space: pre !important;      /* Désactive le retour à la ligne */
    overflow-x: auto !important;      /* Active le slider horizontal */
    overflow-y: auto !important;      /* Active le slider vertical */
    height: 45vh !important;          /* Utilise 65% de la hauteur de l'écran */
}

"""

CSS += visualizer.css(config.get("visualizer"))
CSS += mic_toggle.css()

with gr.Blocks(css=CSS, title="Aria Voice") as ariaHmi:


    with gr.Row(equal_height=True): # Force les colonnes à avoir la même hauteur
        with gr.Column(scale=1):
            gr.Markdown("# 🎙️ Aria Voice System")
            gr.Markdown(
                """
                **Bienvenue sur l'interface de contrôle d'Aria !**  
                Utilisez ce panneau pour configurer les différents modules (VAD, STT, LLM, TTS) et visualiser les logs en temps réel.  
                Cliquez sur le cadre pour activer ou couper le microphone et le son.  
                """
            )
        with gr.Column(scale=4, elem_id="visualizer_col"): # Visualiseur audio : anneaux animés par la voix d'Aria
            gr.HTML('<canvas id="audio_visualizer" width="400" height="100"></canvas>', elem_id="visualizer_wrapper")
            # Interlocuteur : qui parle à Aria. Masquée tant que le serveur de mémoire n'est pas connecté (voir refresh_interlocutor).
            interlocutor_dd = gr.Dropdown(choices=[("Inconnue", "")], value="", show_label=False, container=False,
                                          interactive=True, allow_custom_value=False, elem_id="interlocutor_dd", visible=False)


    with gr.Tabs():

        # --- ONGLET PRINCIPAL (CHAT) ---
        with gr.TabItem("💬 Interface Chat"):
            
            chatbot = gr.Chatbot(elem_id="aria_chatbot")
            with gr.Row():
                text_input = gr.Textbox(elem_id="aria_text_input", show_label=False, lines=1, max_lines=4, scale=8,
                                        placeholder="Écrire un message à Aria puis Entrée (en plus de la voix)")
                stop_btn = gr.Button("⏹ Stop", elem_id="aria_stop_btn", variant="stop", scale=1, min_width=90)
                new_btn = gr.Button("🧹 Nouvelle conversation", elem_id="aria_new_btn", scale=2, min_width=150)
            
            # Bridge technique (visibles pour le DOM mais cachés par CSS)
            audio_url_bridge = gr.Textbox(elem_id="audio_url_bridge", visible=True)
            audio_input = gr.Textbox(elem_id="audio_input_box", visible=True)
            trigger_btn = gr.Button("Trigger", elem_id="aria_trigger", visible=True)
            
            # with gr.Row():
            #     start_btn = gr.Button("🚀 ACTIVER MICRO & SON", variant="primary")
                
            # start_btn.click(None, None, None, js=JS_COMBO)
            
        # --- NOUVEL ONGLET DE GESTION DES CONFIGS ---
        # --- ONGLET DE GESTION DES CONFIGS ---
        with gr.TabItem("⚙️ Paramètres"):
            # Boutons en haut de l'onglet (leurs actions sont reliées en bas, une fois tous les champs créés)
            with gr.Row():
                save_btn = gr.Button("💾 SAUVEGARDER TOUTE LA CONFIGURATION", variant="primary", scale=4)
                restart_btn = gr.Button("🔄 Redémarrer", variant="stop", scale=1)
            status_msg = gr.Markdown("")
            gr.Markdown("---")

            gr.Markdown("### 🛠️ Configuration Modulaire")

            # --- BLOC 0 : RÉGLAGES GÉNÉRAUX (langue de tous les modules) ---
            with gr.Group():
                gr.Markdown("#### 🌍 Réglages généraux")
                gr.Markdown("<small>`language` : code de la langue de toute la chaîne (STT, LLM, TTS), ex. `fr`, `en`, `es`. "
                            "`default_language` : langue de repli si un module ne gère pas la langue demandée. "
                            "Les modules actifs se choisissent plus bas.</small>")
                global_code = gr.Code(
                    value=json.dumps(config.get("global", {}), indent=2, ensure_ascii=False),
                    language="json",
                    label="Réglages généraux",
                    lines=4
                )
            
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

            # --- BLOC 1b : CONVERSATION (mots d'ouverture, de clôture et d'arrêt) ---
            with gr.Group():
                gr.Markdown("#### 💬 Conversation")
                gr.Markdown("<small>`enabled` : active les sessions (Aria n'écoute que pendant une conversation, ouverte par un mot de "
                            "`open_phrases` en début de phrase, fermée par un mot de `close_phrases`). `stop_phrases` : mots qui coupent "
                            "Aria. Accents, majuscules et ponctuation sont ignorés.</small>")
                conv_code = gr.Code(
                    value=json.dumps(config.get("conversation", {}), indent=2, ensure_ascii=False),
                    language="json",
                    label="Paramètres de la conversation",
                    lines=8
                )
                gr.Markdown("#### 🧠 Mémoire (historique envoyé au LLM)")
                gr.Markdown("<small>`reset_on_close` : la clôture d'une conversation efface l'historique. `max_exchanges` et `max_chars` : "
                            "plafonds (les plus anciens échanges sont retirés). `tool_result_max_chars` : taille à laquelle un ancien "
                            "résultat d'outil est raccourci. 0 = sans limite.</small>")
                memory_code = gr.Code(
                    value=json.dumps(config.get("memory", {}), indent=2, ensure_ascii=False),
                    language="json",
                    label="Paramètres de la mémoire",
                    lines=6
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

                # --- BLOC 5 : MCP (Outils & Serveurs) ---
            with gr.Group():
                gr.Markdown("#### 🛠️ Outils MCP (Model Context Protocol)")
                gr.Markdown("<small>Configurez ici vos serveurs d'outils (Météo, Bases de données, etc.)</small>")
                
                mcp_code = gr.Code(
                    value=json.dumps(config["mcp"]["params"], indent=2),
                    language="json",
                    label="Configuration des serveurs MCP",
                    lines=12
                )

            # --- BLOC 6 : AGENTS (profils des clés API) ---
            with gr.Group():
                gr.Markdown("#### 🤖 Agents : profils")
                profiles_help_md = gr.Markdown(settings_mod.profiles_help(config))
                profiles_code = gr.Code(
                    value=json.dumps(agents_compat.section(config).get("profiles", {}), indent=2, ensure_ascii=False),
                    language="json",
                    label="Profils des agents (chaque agent en choisit un)",
                    lines=12
                )
                close_open_cb = gr.Checkbox(label="Fermer les conversations ouvertes des clés dont le profil change (sinon, elles gardent l'ancien jusqu'à leur expiration)",
                                            value=False)

            # --- BLOC 7 : AUDIT (journal des échanges) ---
            with gr.Group():
                gr.Markdown("#### 🧾 Audit des échanges")
                gr.Markdown("<small>Journal de chaque échange (question, réponse, source, interlocuteur) dans <code>models/audit/</code>, hors git. "
                            "<code>retention_days</code> : conservation par défaut, purge automatique (une clé d'accès peut exiger moins). "
                            "<code>log_text</code> : false = empreinte à la place du texte. Le dossier (<code>dir</code>) n'est pris en compte qu'au redémarrage. "
                            "Purge manuelle et journal des purges : page d'administration des accès.</small>")
                audit_code = gr.Code(
                    value=json.dumps({"enabled": True, "retention_days": echanges.DEFAULT_RETENTION_DAYS, "log_text": True, "dir": "models/audit",
                                      **(config.get("audit") or {})}, indent=2, ensure_ascii=False),
                    language="json",
                    label="Journal d'audit",
                    lines=6
                )

            # --- SAUVEGARDE GLOBALE (boutons en haut de l'onglet) ---
            # Au clic, on envoie toutes les valeurs actuelles (dropdowns + codes)
            save_btn.click(
                fn=update_config_ui,
                inputs=[
                    global_code,
                    vad_dd, vad_code,
                    conv_code,
                    memory_code,
                    stt_dd, stt_code,
                    llm_dd, llm_code,
                    tts_dd, tts_code,
                    mcp_code,
                    profiles_code, close_open_cb, audit_code
                ],
                outputs=[status_msg]
            )

            # Redémarrage : le processus se termine et la politique « restart: unless-stopped » du conteneur le relance
            # Champ texte caché par le CSS (comme les ponts audio) : un gr.State reste côté serveur,
            # le navigateur ne peut pas lui envoyer la réponse de la boîte de confirmation
            restart_confirm = gr.Textbox(elem_id="restart_confirm_box", visible=True, value="")
            restart_btn.click(
                fn=request_restart,
                inputs=[restart_confirm],
                outputs=[status_msg],
                js=JS_RESTART
            )

        # --- ONGLET LOGS (RAM) ---
        with gr.TabItem("📟 Logs Serveur"):
            
            with gr.Row():
                    gr.Markdown("### Flux de logs en temps réel")
                    auto_refresh = gr.Checkbox(label="Auto-refresh (1s)", value=True)
                    refresh_logs_btn = gr.Button("🔄 Actualiser", size="sm")

            # Utilisation de Textbox pour l'autoscroll et la mise en page
            logs_output = gr.Textbox(
                value="En attente d'activité...",
                label=None, 
                lines=25,          
                max_lines=25,       # Doit être égal ou proche de lines pour forcer le scroll
                interactive=False,
                elem_id="log_display",
                autoscroll=True     # Garde le focus sur la dernière ligne
            )
            
            # Timer : Déclenche la mise à jour toutes les secondes
            timer = gr.Timer(1)
            
            # Connexion du Timer à la lecture mémoire
            timer.tick(
                fn=read_logs_memory,
                inputs=None,
                outputs=[logs_output],
                show_progress="hidden" # Cache la barre de chargement gênante
            )
            
            # Gestion du checkbox pour activer/désactiver le timer
            auto_refresh.change(
                fn=lambda x: gr.Timer(active=x),
                inputs=[auto_refresh],
                outputs=[timer]
            )
            
            refresh_logs_btn.click(fn=read_logs_memory, outputs=[logs_output])

        # --- ONGLET MÉMOIRE (page de gestion du serveur de mémoire) : seulement si elle est activée dans la config ---
        _memory_port = memory_tab.ui_port(config)
        if _memory_port:
            with gr.TabItem("🧠 Mémoire") as memory_tab_item:
                gr.HTML(memory_tab.html(_memory_port))
            memory_tab_item.select(None, None, None, js=memory_tab.js_reload())

        # --- ONGLET AGENTS (clés API des agents) : seulement si "agents.enabled" est vrai dans la config ---
        _agents_port = agents_tab.admin_port(config)
        if _agents_port:
            with gr.TabItem("🤖 Agents") as agents_tab_item:
                gr.HTML(agents_tab.html(_agents_port))
            agents_tab_item.select(None, None, None, js=agents_tab.js_reload())

    ariaHmi.load(None, None, None, js=JS_ON_LOAD)
    # Au chargement de la page : liste à jour (ou masquée si le serveur de mémoire est absent)
    ariaHmi.load(route_refresh_interlocutor, None, [interlocutor_dd], show_progress="hidden")

    text_input.submit(
        fn=route_process_texte,
        inputs=[text_input, chatbot, interlocutor_dd],
        outputs=[chatbot, audio_url_bridge, text_input],
        show_progress="hidden",
        concurrency_limit=None
    ).then(route_refresh_interlocutor, None, [interlocutor_dd], show_progress="hidden")

    # Changement à la main : le serveur de mémoire est prévenu (et l'historique effacé si on passe d'une personne à une autre)
    interlocutor_dd.input(route_change_interlocutor, [interlocutor_dd], None, show_progress="hidden")

    stop_btn.click(
        fn=route_stop_aria,
        inputs=None,
        outputs=[audio_url_bridge],
        show_progress="hidden",
        concurrency_limit=None,
        js="() => { if (window.ariaStopAudio) window.ariaStopAudio(); }"     # coupe le son tout de suite, sans attendre le serveur
    )

    new_btn.click(
        fn=route_new_conversation,
        inputs=None,
        outputs=[chatbot, audio_url_bridge],
        show_progress="hidden",
        concurrency_limit=None,
        js="() => { if (window.ariaStopAudio) window.ariaStopAudio(); }"
    ).then(route_refresh_interlocutor, None, [interlocutor_dd], show_progress="hidden")

    trigger_btn.click(
        fn=route_process_streaming_binaire,
        inputs=[audio_input, chatbot, interlocutor_dd],
        outputs=[chatbot, audio_url_bridge],
        show_progress="hidden",
        concurrency_limit=None
    ).then(route_refresh_interlocutor, None, [interlocutor_dd], show_progress="hidden")

if __name__ == "__main__":
    import asyncio
    import contextlib

    @contextlib.asynccontextmanager
    async def _mcp_lifespan(app):
        """Connecte les serveurs MCP au démarrage d'Aria (et non à la première question) : leurs pages éventuelles
        (ex. gestion de la mémoire) sont disponibles tout de suite, et le premier échange ne paie plus ce démarrage."""
        manager = getattr(llm.engine, "mcp_manager", None)
        _boucle_mcp["loop"] = asyncio.get_running_loop()
        _boucle_interface.set(_boucle_mcp["loop"])                      # pont de l'agent `aria` (components/agents/bridge.py) : ses tours s'exécutent ici

        async def connecter():
            try:
                await manager.ensure_connected()
            except Exception as e:                  # ne jamais empêcher Aria de démarrer à cause d'un serveur MCP
                utils.log_info("MCP", f"❌ Connexion au démarrage impossible : {type(e).__name__}: {e}")

        # La connexion se fait en tâche de fond : Gradio n'attend que 5 s que le serveur soit prêt, et le lifespan le retarde
        # tant qu'il n'a pas rendu la main (jeedom, recherche web, mémoire : plusieurs secondes). Un message qui arrive avant la fin
        # attend le verrou de ensure_connected, comme avant.
        tache = asyncio.create_task(connecter()) if manager is not None else None
        try:
            yield
        finally:
            if tache is not None and not tache.done():
                tache.cancel()
            if manager is not None:
                await manager.close()

    # Droit à l'effacement (page d'administration des accès, thread HTTP) : l'outil caché de la mémoire est appelé dans la boucle de l'application
    def _effacer_traces_memoire(qui, ids):
        manager, loop = mcp_manager(), _boucle_mcp.get("loop")
        if manager is None or loop is None or not manager.has_tool("ui_effacer_traces"):
            return {"erreur": "mémoire indisponible"}
        texte = asyncio.run_coroutine_threadsafe(manager.call_tool("ui_effacer_traces", {"qui": qui, "echanges": ",".join(ids[:500])}), loop).result(20)
        return json.loads(texte)

    # HTTPS optionnel (config "https") : nécessaire pour le micro depuis un autre poste ou un smartphone
    _https_args = https.launch_args(config.get("https"))
    # Agents (applications ou appareils qui dialoguent avec Aria par clé API ; config "agents", désactivé par défaut) : API sur un port dédié
    try:
        from components.agents import server as agents_server
        # Les agents sont TOUJOURS assemblés (l'agent par défaut `aria` sert le chat) ; le port 7861 n'est ouvert que si `agents.enabled`.
        # `shared_engine` : l'agent `aria` réutilise le Llm et le MCP globaux d'Aria (interlocuteur global, mémoire du foyer).
        agents_srv = agents_server.start(config, stt, tts, _https_args, language=language, effacer_traces=_effacer_traces_memoire,
                                         shared_engine=lambda: (llm, getattr(llm.engine, "mcp_manager", None)), loop_holder=_boucle_interface)
        from components.agents import chat as agents_chat
        chat_aria = agents_chat.ChatAria(agents_srv, config, mcp_manager, tts=tts)
    except Exception as e:                          # jamais empêcher Aria de démarrer à cause de cette option
        utils.log_info("AGENTS", f"❌ API des agents non démarrée : {type(e).__name__}: {e}")
    # Purge automatique du journal d'audit (au démarrage puis toutes les 6 h) : durée par défaut et durée de chaque accès relues à chaque passage
    def _reglages_audit():
        cles = getattr(getattr(agents_srv, "agents_deps", None), "keys", None)
        jours = (config.get("audit") or {}).get("retention_days", echanges.DEFAULT_RETENTION_DAYS)
        return jours, (cles.retention_of if cles else None)
    try:
        audit_retention.Planificateur(_reglages_audit).start()
    except Exception as e:
        utils.log_info("AUDIT", f"❌ Purge automatique non démarrée : {type(e).__name__}: {e}")
    ariaHmi.launch(server_name="0.0.0.0", server_port=7860, app_kwargs={"lifespan": _mcp_lifespan}, **_https_args)