"""Micro/son d'Aria commandés par un clic sur le cadre du haut (plus de bouton « Activer Micro & Son »).

États (classe CSS sur #audio_visualizer, image affichée) :
  1. off        micro et son coupés            -> muted_mic.gif, fixe (première image)
  2. listening  Aria écoute                     -> transition.gif
  3. hearing    le VAD détecte une phrase       -> loading.gif
  4. speaking   Aria répond, courbes affichées  -> (courbes dessinées par static/audioDesign.js)
  5. retour à 2 à la fin de la réponse

Un clic sur le cadre, à n'importe quel état, bascule entre « coupé » et « écoute ».
Module isolé : app.py n'a qu'à ajouter css() au CSS, et js_toggle() au JavaScript du clic.
"""
import base64
import pathlib
import components.utils as utils

STATIC = pathlib.Path(__file__).resolve().parent.parent / "static"

# Image de chaque état. « muted_mic_still.png » est la 1re image de muted_mic.gif (une image fixe, sans animation).
IMAGES = {
    "off": "muted_mic_still.png",
    "listening": "transition.gif",
    "hearing": "loading.gif",
}


def _data_uri(name):
    """Image en base64 intégrée à la page : Gradio ne sert pas le dossier static/ (404/403)."""
    path = STATIC / name
    try:
        raw = path.read_bytes()
    except OSError as e:
        utils.log_info("MIC", f"Image absente ({name}) : {e}")
        return ""
    mime = "image/png" if name.endswith(".png") else "image/gif"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def css():
    """Une image de fond par état, le canvas des courbes n'est visible que pendant la réponse."""
    rules = []
    for state, name in IMAGES.items():
        uri = _data_uri(name)
        if uri:
            rules.append(f'#audio_visualizer.state-{state} {{ background-image: url("{uri}"); }}')
    return """
/* --- Micro/son par clic sur le cadre (components/mic_toggle.py) --- */
#audio_visualizer { cursor: pointer; background-repeat: no-repeat; background-position: center; background-size: contain; background-color: transparent; }
""" + "\n".join(rules) + """
/* Pendant la réponse (speaking) : seules les courbes. Au repos hors conversation (idle) : cadre vide */
#audio_visualizer.state-speaking, #audio_visualizer.state-idle { background-image: none; }
"""


def js_toggle(init_js, conversation=False):
    """JavaScript de l'interrupteur et des transitions d'état (exécuté au chargement de la page).
    conversation : sessions de conversation activées -> pas d'icône tant qu'aucune conversation n'est ouverte.
    init_js : le JavaScript d'initialisation du son (VAD + lecture + visualiseur), exécuté au premier clic seulement."""
    return ("window.ariaConvEnabled = " + ("true" if conversation else "false") + ";" + """
window.ariaInitAudio = async () => {
    if (window.ariaAudioReady) {
        // Réactivation après une coupure : le VAD existant reprend (ou est recréé si le micro a été libéré)
        if (window.myVAD) { window.myVAD.start(); return; }
    }
    window.ariaAudioReady = true;
    await (""" + init_js + """);
    if (window.ariaVadPromise) await window.ariaVadPromise;
    if (!window.myVAD) throw new Error('VAD non initialisé (micro refusé ?)');
};
(() => {
    if (window.ariaMic) return;
    const box = () => document.getElementById('audio_visualizer');
    const A = window.ariaMic = { state: 'off', ready: false, busy: false, convOpen: false, pendingClose: false };
    // État de repos du cadre : aucune icône (idle) tant qu'aucune conversation n'est ouverte, sinon écoute
    A.rest = () => (window.ariaConvEnabled && !A.convOpen) ? 'idle' : 'listening';

    A.set = (s) => {
        A.state = s;
        const el = box();
        if (!el) return;
        el.classList.remove('state-off', 'state-idle', 'state-listening', 'state-hearing', 'state-speaking');
        el.classList.add('state-' + s);
        el.title = s === 'off' ? 'Cliquer pour activer le micro et le son' : 'Cliquer pour couper le micro et le son';
    };

    // Micro réellement coupé : on arrête les pistes du flux (l'icône du navigateur s'éteint), puis on rétablit au clic suivant
    A.stopMic = () => {
        try { if (window.myVAD) window.myVAD.pause(); } catch (e) {}
        try { if (window.myVAD && window.myVAD.stream) window.myVAD.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
        try { if (window.myVAD && window.myVAD.audioContext) window.myVAD.audioContext.suspend(); } catch (e) {}
    };
    A.startMic = async () => {
        // Le flux a été arrêté : on détruit l'ancien VAD pour qu'il soit recréé avec un micro neuf
        if (window.myVAD && window.myVAD.stream && !window.myVAD.stream.active) {
            try { window.myVAD.audioContext && window.myVAD.audioContext.close(); } catch (e) {}
            window.myVAD = null; window.ariaInitialized = false;
        }
        await window.ariaInitAudio();
    };

    // Transitions pilotées par le VAD et par la lecture. Sans effet quand le micro est coupé.
    A.onPhrase = () => { if (A.state === 'listening') A.set('hearing'); };                         // 2 -> 3
    window.addEventListener('aria_audio_start', () => { if (A.state !== 'off') A.set('speaking'); });  // 3 -> 4
    window.addEventListener('aria_audio_end', () => {                                                // 4 -> 2
        if (A.pendingClose) { A.pendingClose = false; A.convOpen = false; }   // clôture reçue pendant la réponse : appliquée à sa fin
        if (A.state === 'speaking' || (A.state !== 'off' && A.state !== 'hearing')) A.set(A.rest());
    });
    // Ouverture / clôture de conversation décidées par le serveur (mots-clés dits ou écrits)
    window.addEventListener('aria_conv_open', () => { A.convOpen = true; A.pendingClose = false; if (A.state !== 'off') A.set('hearing'); });
    window.addEventListener('aria_conv_close', () => {
        if (window.isPlaying) { A.pendingClose = true; return; }
        A.convOpen = false; if (A.state !== 'off') A.set(A.rest());
    });
    // Phrase ignorée ou réponse arrêtée : retour au repos sans attendre le délai de sécurité
    window.addEventListener('aria_idle', () => { if (A.state === 'hearing' || A.state === 'speaking') A.set(A.rest()); });
    // Le serveur n'a rien à dire (phrase vide ignorée, erreur) : on revient à l'écoute au lieu de rester sur « chargement »
    A.watchdog = setInterval(() => {
        if (A.state === 'hearing' && Date.now() - (A.hearingSince || 0) > 60000) A.set(A.rest());
    }, 5000);
    const _set = A.set;
    A.set = (s) => { if (s === 'hearing') A.hearingSince = Date.now(); _set(s); };

    A.toggle = async () => {
        if (A.busy) return;
        A.busy = true;
        try {
            if (A.state === 'off') {
                await A.startMic();
                window.muteAria = false;
                A.set(A.rest());
            } else {
                // Coupe aussi la lecture en cours et vide la file : « désactivé » veut dire silence immédiat
                window.muteAria = true;
                window.audioQueue = []; window.isPlaying = false;
                try { (window.ariaCurrentAudio || {}).pause && window.ariaCurrentAudio.pause(); } catch (e) {}
                A.stopMic();
                A.set('off');
            }
        } catch (e) {
            console.error('[ARIA-MIC] Échec du basculement :', e);
            A.set('off');
        } finally { A.busy = false; }
    };

    const el = box();
    if (el) { el.addEventListener('click', A.toggle); A.set('off'); }
    else console.warn('[ARIA-MIC] cadre #audio_visualizer introuvable');
})();
""")
