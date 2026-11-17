"""Liaison d'un accès tiers avec le serveur de mémoire UNIQUE d'Aria (« memoire »), par son canal HTTP local.

Un profil référence le serveur `memoire` comme n'importe quel autre ; si ce serveur déclare `MEMORY_HTTP_PORT`, une session tierce ne lance pas son propre
processus : elle se connecte au canal HTTP du processus d'Aria, avec
  - `Authorization: Bearer <secret>`  le secret du canal, généré à CHAQUE démarrage d'Aria, qui n'existe qu'en mémoire (jamais écrit dans un fichier
                                      ni dans la config) ; il est transmis au processus mémoire par son environnement ;
  - `X-Memory-Principal: <clé interne de l'accès>`  l'identité de la personne système ; le serveur l'impose, le LLM n'y a pas accès ;
  - `X-Memory-Name: <nom de l'accès>`  son nom (encodé), pour que la fiche suive un renommage.
Le LLM d'un tiers ne reçoit jamais `identifier_interlocuteur` (le serveur ne le sert pas non plus : double sécurité).
"""
import json
import secrets
from urllib.parse import quote

HTTP_HOST = "127.0.0.1"
SECRET = {"value": ""}           # secret du canal pour CE démarrage d'Aria (mémoire vive seulement)
HIDDEN_FOR_THIRD_PARTIES = ("identifier_interlocuteur",)


def new_secret():
    """Génère le secret de ce démarrage (48 caractères) et le conserve en mémoire vive. Un nouvel appel le remplace."""
    SECRET["value"] = secrets.token_urlsafe(36)
    return SECRET["value"]


def current_secret():
    return SECRET["value"]


def http_port(conf):
    """Port du canal HTTP déclaré dans la configuration d'un serveur MCP (`env.MEMORY_HTTP_PORT`), ou None."""
    try:
        port = int(((conf or {}).get("env") or {}).get("MEMORY_HTTP_PORT") or 0)
    except (TypeError, ValueError):
        return None
    return port if 0 < port < 65536 else None


def is_memory_server(conf):
    """Un serveur est un serveur de mémoire (au sens de ce module) s'il déclare un canal HTTP."""
    return http_port(conf) is not None


def with_secret(conf, secret=None):
    """Configuration du serveur mémoire pour le processus d'Aria : le secret du canal est ajouté à son environnement (sans modifier la config)."""
    port = http_port(conf)
    if port is None:
        return conf
    secret = secret or current_secret()
    if not secret:
        return conf
    env = {**(conf.get("env") or {}), "MEMORY_HTTP_SECRET": secret}
    return {**conf, "env": env}


def third_party_conf(conf, key_id, name, secret=None, publish=False, speaker=None, session=None):
    """Configuration de connexion d'un serveur mémoire pour UNE session tierce : URL locale du canal HTTP et en-têtes d'identité.
    Retourne None si le serveur n'a pas de canal HTTP ou si le secret du démarrage n'est pas disponible (la mémoire est alors indisponible,
    jamais ouverte sans authentification)."""
    port = http_port(conf)
    secret = secret or current_secret()
    if port is None or not secret or not key_id or not str(name or "").strip():
        return None
    headers = {"Authorization": f"Bearer {secret}", "X-Memory-Principal": str(key_id), "X-Memory-Name": quote(str(name).strip(), safe="")}
    if publish:
        headers["X-Memory-Publish"] = "1"          # option de la clé : ses liens avec une personne sont publics ; posé par Aria, jamais par le LLM
    if speaker is not None:
        # Accès qui publie les informations de son interlocuteur : canal /mcp-i, interlocuteur propre à la session (fourni par l'application s'il l'est).
        headers["X-Memory-Session"] = quote(str(session or "default"), safe="")
        given = speaker or {}
        if given.get("n"):
            headers["X-Memory-Speaker"] = quote(json.dumps({"n": str(given["n"])[:60], "c": str(given.get("c", ""))[:60]}, ensure_ascii=False, separators=(",", ":")), safe="")
        return {"url": f"http://{HTTP_HOST}:{port}/mcp-i", "headers": headers, "hide_tools": [] if not given.get("n") else ["identifier_interlocuteur"]}
    return {"url": f"http://{HTTP_HOST}:{port}/mcp", "headers": headers, "hide_tools": list(HIDDEN_FOR_THIRD_PARTIES)}


def resolve_servers(servers, key_id, name, secret=None, publish=False, speaker=None, session=None):
    """Serveurs MCP d'une session tierce : un serveur mémoire est remplacé par sa connexion HTTP identifiée, les autres sont inchangés.
    Retourne (serveurs résolus, noms des serveurs mémoire indisponibles)."""
    out, missing = {}, []
    for srv_name, conf in servers.items():
        if is_memory_server(conf):
            link = third_party_conf(conf, key_id, name, secret, publish, speaker, session)
            if link is None:
                missing.append(srv_name)
            else:
                out[srv_name] = link
        else:
            out[srv_name] = conf
    return out, missing
