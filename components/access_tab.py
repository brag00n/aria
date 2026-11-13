"""Onglet « Accès tiers » : intègre la page d'administration des clés API (components/access/admin_ui.py) dans l'interface d'Aria.

L'onglet n'existe que si `access.enabled` est vrai dans la config. La page est servie sur un port à part, publié sur 127.0.0.1 seulement
(docker-compose.yml) : le port 7860 de l'interface est ouvert au réseau local sans authentification, la création de clés n'y est donc
jamais exposée.
"""
FRAME_ID = "aria_access_frame"


def admin_port(config):
    """Port de la page d'administration si les accès tiers sont activés, sinon None."""
    access = (config or {}).get("access") or {}
    if not access.get("enabled"):
        return None
    try:
        port = int(access.get("admin_port", 7871))
    except (TypeError, ValueError):
        return None
    return port if 0 < port < 65536 else None


def html(port):
    url = f"http://127.0.0.1:{port}/"
    return (
        f'<div style="display:flex;gap:12px;align-items:center;margin-bottom:6px">'
        f'<a href="{url}" target="_blank" rel="noopener">Ouvrir dans un onglet ↗</a>'
        f'<small style="opacity:.7">Page servie par Aria ({url}). Visible depuis ce PC uniquement.</small></div>'
        f'<iframe id="{FRAME_ID}" data-src="{url}" title="Accès tiers" '
        f'style="width:100%;height:78vh;border:1px solid var(--border-color-primary,#ccc);border-radius:8px;background:#fff"></iframe>'
    )


def js_reload():
    return ("() => { const f = document.getElementById('" + FRAME_ID + "'); "
            "if (f) f.src = f.dataset.src + '?t=' + Date.now(); }")
