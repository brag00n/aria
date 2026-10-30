"""Onglet « Mémoire » : intègre la page de gestion du serveur de mémoire (tools/memory/ui.py) dans l'interface d'Aria.

Module isolé : l'onglet n'existe que si l'entrée MCP « memoire » définit MEMORY_UI_PORT. Sans serveur de mémoire,
rien n'est affiché et app.py n'a rien d'autre à savoir. La page elle-même est servie par le serveur de mémoire (autre port).
"""

SERVER_NAME = "memoire"
FRAME_ID = "aria_memory_frame"


def ui_port(config):
    """Port de la page de gestion si elle est activée dans la config, sinon None."""
    try:
        env = config["mcp"]["params"]["servers"][SERVER_NAME].get("env") or {}
        port = int(env.get("MEMORY_UI_PORT") or 0)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return port if 0 < port < 65536 else None


def html(port):
    """Cadre intégré. Adresse 127.0.0.1 : le port n'est publié que sur la machine hôte (docker-compose.yml).
    Le cadre n'est chargé qu'à l'ouverture de l'onglet (data-src), pour afficher des souvenirs à jour."""
    url = f"http://127.0.0.1:{port}/"
    return (
        f'<div style="display:flex;gap:12px;align-items:center;margin-bottom:6px">'
        f'<a href="{url}" target="_blank" rel="noopener">Ouvrir dans un onglet ↗</a>'
        f'<small style="opacity:.7">Page servie par le serveur de mémoire ({url}). '
        f'Visible depuis ce PC uniquement.</small></div>'
        f'<iframe id="{FRAME_ID}" data-src="{url}" title="Mémoire d\'Aria" '
        f'style="width:100%;height:78vh;border:1px solid var(--border-color-primary,#ccc);border-radius:8px;background:#fff"></iframe>'
    )


def js_reload():
    """JavaScript exécuté quand l'onglet s'ouvre : (re)charge la page pour afficher les souvenirs à jour."""
    return ("() => { const f = document.getElementById('" + FRAME_ID + "'); "
            "if (f) f.src = f.dataset.src + '?t=' + Date.now(); }")
