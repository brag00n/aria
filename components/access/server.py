"""Démarrage de l'API tierce dans Aria : serveur uvicorn dans un thread, port dédié, même certificat que l'interface.

Section de configuration (absente ou `enabled: false` = fonction désactivée, rien n'est ouvert) :
    "access": {"enabled": true, "port": 7861, "max_sessions": 8, "session_ttl_s": 900, "data_dir": "models/api_access",
               "private_servers": [], "profiles": {...}}
Les profils (outils MCP, message système, LLM par clé) sont dans `access.profiles` : voir components/access/profile.py. Ils sont relus à chaque
création de session, sans redémarrage.
"""
import asyncio
import io
import threading

import components.utils as utils

DEFAULTS = {"enabled": False, "port": 7861, "admin_port": 7871, "max_sessions": 8, "session_ttl_s": 900,
            "data_dir": "models/api_access"}


def settings(config):
    s = dict(DEFAULTS)
    s.update((config or {}).get("access") or {})
    try:
        s["port"] = int(s["port"])
        s["admin_port"] = int(s["admin_port"])
        s["max_sessions"] = max(1, int(s["max_sessions"]))
        s["session_ttl_s"] = max(30, int(s["session_ttl_s"]))
    except (TypeError, ValueError):
        s.update({k: DEFAULTS[k] for k in ("port", "admin_port", "max_sessions", "session_ttl_s")})
    return s


def make_decoder():
    """Décodage d'un audio quelconque (wav du navigateur) en tableau float32 mono 16 kHz, comme l'interface d'Aria."""
    def decode(raw):
        import numpy as np
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(raw)).set_frame_rate(16000).set_channels(1)
        full_scale = float(1 << (8 * seg.sample_width - 1))
        return np.array(seg.get_array_of_samples()).astype(np.float32) / full_scale
    return decode


def build(config, stt, tts, language=None):
    """Assemble les dépendances réelles. Retourne (app Starlette, deps, store de clés, profils)."""
    from . import api, keys, profile, sessions
    s = settings(config)
    store = keys.KeyStore(utils.aria_path(*s["data_dir"].split("/")))
    profiles = profile.ConfigProfiles(lambda: config)          # config VIVANTE : les profils modifiés dans l'onglet Paramètres valent tout de suite
    registry = sessions.SessionRegistry(profiles, lambda prof, identity=None: sessions.build_engine(prof, config, language, identity),
                                        max_sessions=s["max_sessions"], ttl_s=s["session_ttl_s"])

    def audit(event, **fields):
        utils.log_info("ACCESS", f"{event} " + " ".join(f"{k}={v}" for k, v in fields.items()))

    deps = api.Deps(keys=store, sessions=registry, stt=stt, tts=tts, decode_audio=make_decoder(), on_audit=audit,
                    static_dir=utils.aria_path("static"))
    return api.create_app(deps), deps, store, profiles


def start(config, stt, tts, https_args, language=None, effacer_traces=None):
    """Lance l'API dans un thread démon si `access.enabled`. Retourne l'objet serveur uvicorn (ou None)."""
    s = settings(config)
    if not s["enabled"]:
        return None
    import uvicorn
    app, deps, store, profiles = build(config, stt, tts, language)
    admin = None
    if s["admin_port"]:
        from . import admin_ui
        try:
            admin, _token = admin_ui.start(store, profiles, deps.sessions, s["admin_port"], audit=deps.on_audit, effacer_traces=effacer_traces)
            utils.log_info("ACCESS", f"Page d'administration sur le port {s['admin_port']} (publié sur 127.0.0.1 seulement)")
        except OSError as e:
            utils.log_info("ACCESS", f"❌ Page d'administration non démarrée : {e}")
    conf = {"host": "0.0.0.0", "port": s["port"], "log_level": "warning", "access_log": False}
    if https_args.get("ssl_certfile"):
        conf.update(ssl_certfile=https_args["ssl_certfile"], ssl_keyfile=https_args["ssl_keyfile"])
    server = uvicorn.Server(uvicorn.Config(app, **conf))

    async def run():
        deps.sessions.start_reaper()
        try:
            await server.serve()
        finally:
            store.flush()
            await deps.sessions.close_all()

    threading.Thread(target=lambda: asyncio.run(run()), name="aria-access", daemon=True).start()
    scheme = "https" if conf.get("ssl_certfile") else "http"
    utils.log_info("ACCESS", f"API des applications tierces sur {scheme}://0.0.0.0:{s['port']}/v1/")
    server.admin = admin
    server.access_deps = deps          # registre des sessions et clés, pour l'onglet Paramètres (fermeture des sessions d'un profil modifié)
    return server


def close_sessions_for_profiles(server, names):
    """Ferme les conversations ouvertes des clés dont le profil figure dans `names` (profil modifié ou supprimé) : elles repartent avec le
    profil à jour à leur prochain tour. Retourne le nombre de conversations fermées. Sans serveur d'accès, ne fait rien."""
    deps = getattr(server, "access_deps", None)
    if deps is None or not names:
        return 0
    names = set(names)
    keys = [k["id"] for k in deps.keys.list() if k.get("profile") in names]
    closed = sum(1 for sid, sess in list(deps.sessions.sessions.items()) if sess.key_id in keys)
    for kid in keys:
        deps.sessions.request_close_key(kid)          # sûr depuis un autre thread : planifié dans la boucle de l'API
    return closed


def keys_using_profiles(server, names):
    """Libellés des clés qui utilisent l'un des profils `names` (pour avertir avant de supprimer un profil utilisé)."""
    deps = getattr(server, "access_deps", None)
    if deps is None:
        return []
    names = set(names)
    return [f"{k['label']} ({k['profile']})" for k in deps.keys.list() if k.get("profile") in names]
