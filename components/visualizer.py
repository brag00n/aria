"""Visualiseur audio du navigateur : anneaux animés par la voix d'Aria, dans un cadre visible en haut de la page.

Module isolé : app.py n'a qu'à ajouter css() à son CSS et build_js() à son JavaScript d'initialisation.
Le dessin est dans static/audioDesign.js ; le canvas #audio_visualizer est créé par app.py.

Réglages (section "visualizer" de configs/default.json, facultatifs) :
  maximize_in_frame : true = les courbes remplissent tout le cadre (largeur et hauteur)
                      false = cercle centré dans le carré
  frame_size_px     : côté du cadre carré, centré en haut de la page, en pixels (40 à 400)
"""
import json
import pathlib
import components.utils as utils

JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "static" / "audioDesign.js"

DEFAULTS = {"maximize_in_frame": True, "frame_size_px": 220}

# Preset "Test003" (celui de la branche WebHMI-tmp)
VISUAL_PARAMS = {"nbRings": 9, "resolution": 30, "spacing": 0, "gain": 3.4, "stroke": 6.6, "alphaMin": 0.0, "alphaMax": 0.74}


def settings(cfg=None):
    """Réglages effectifs : valeurs par défaut, complétées par la config, avec contrôle des types."""
    s = dict(DEFAULTS)
    for key, value in (cfg or {}).items():
        if key in s and type(value) is type(s[key]):
            s[key] = value
    s["frame_size_px"] = max(40, min(int(s["frame_size_px"]), 400))
    return s


def css(cfg=None):
    """CSS du cadre : toujours visible, il s'éclaire en rouge quand l'utilisateur parle."""
    h = settings(cfg)["frame_size_px"]
    return f"""
/* --- Visualiseur audio (components/visualizer.py) --- */
/* Aucune marge ni espacement autour du cadre : il occupe toute la hauteur de sa colonne */
#visualizer_col {{ padding: 0 !important; gap: 0 !important; }}
/* Le cadre touche le haut du navigateur : plus de marge au-dessus de la page */
.gradio-container, .gradio-container > .main, .gradio-container .app {{ padding-top: 8px !important; margin-top: 0 !important; }}
#visualizer_wrapper, #visualizer_wrapper > * {{ margin: 0 !important; padding: 0 !important; border: 0 !important; display: flex; justify-content: center; }}
#audio_visualizer {{
    display: block;
    box-sizing: border-box;
    width: {h}px;
    height: {h}px;
    max-width: 100%;
    margin: 0 auto !important;      /* carré centré dans sa colonne */
    border: 2px solid #9aa0a6;
    border-radius: 12px;
    transition: border-color 0.2s ease;
}}
#audio_visualizer.speaking {{
    border-color: #ff4b4b;
    animation: aria-pulse 1.5s infinite;
}}
@keyframes aria-pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }}
    70%  {{ box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }}
}}
"""


def build_js(cfg=None):
    """JavaScript du visualiseur, isolé dans un try/catch : s'il échoue, le VAD et la lecture audio démarrent quand même."""
    s = settings(cfg)
    try:
        code = JS_PATH.read_text(encoding="utf-8")
    except OSError as e:
        utils.log_info("VISUALIZER", f"Visualiseur désactivé : {e}")
        return ""
    params = dict(VISUAL_PARAMS, fitFrame=s["maximize_in_frame"])
    return f"""try {{
        window.visualParams = {json.dumps(params)};
        {code}
        if (!window.ariaVisuListener) {{
            window.ariaVisuListener = true;
            window.addEventListener('aria_audio_start', (e) => {{
                const el = e.detail && e.detail.audioElement;
                if (el && window.setupAriaVisualizer) window.setupAriaVisualizer(el);
            }});
        }}
    }} catch (err) {{ console.error("[ARIA-VISU] Visualiseur désactivé :", err); }}"""
