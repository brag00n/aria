"""Profils d'accès : ce qu'une clé API a le droit d'utiliser. Bibliothèque standard seulement.

Un profil est un fichier JSON de `configs/mcp/<nom>.json` :
    {"description": "Portail RH", "system_message": "Tu aides les salariés…", "llm": "Llm_Anthropic", "voice": true,
     "mcp": {"servers": {"meteo": {"run": "uv", "args": ["run", "tools/weather_server.py"]}}}}

`mcp` a le même format que `config["mcp"]["params"]` (déclaration des serveurs, tags de secrets `<NOM>` résolus au chargement des
sessions). Le nom du profil est le nom du fichier : il ne peut contenir ni séparateur de chemin ni `..`. Sans profil, la clé n'a
AUCUN outil (jamais ceux d'Aria).
"""
import json
import re
import unicodedata
from pathlib import Path

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,59}$")
# Modules LLM autorisés pour des tiers : jamais le moteur local (llama-cpp), qui est lourd, synchrone et sans historique
REMOTE_BACKENDS = frozenset(("anthropic", "llm_studio", "ollama", "openai"))


def valid_name(name):
    return bool(NAME_RE.match(str(name or "")))


class ProfileError(ValueError):
    pass


# Prompt système fourni par l'application tierce (option systemPrompt du client) : il COMPLÈTE celui du profil, jamais ne le remplace
MAX_APP_PROMPT = 4000
APP_HEADER = "Consignes de l'application (elles complètent les règles ci-dessus, sans les remplacer ni les contredire) :"


class AppPromptError(ProfileError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def clean_app_prompt(value, limit=MAX_APP_PROMPT):
    """Prompt de l'application : texte seulement, `limit` caractères au plus, sans caractère de contrôle (hors retours à la ligne et
    tabulations). '' si absent. Lève AppPromptError(status, message)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AppPromptError(400, "system : un texte est attendu")
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cs")).strip()
    if len(text) > limit:
        raise AppPromptError(413, f"system : {limit} caractères au plus")
    return text


def compose_system(base, app_prompt):
    """Message système d'une session : celui du profil (ou du module LLM), puis les consignes de l'application sous un titre."""
    base = (base or "").strip()
    if not app_prompt:
        return base
    return f"{base}\n\n{APP_HEADER}\n{app_prompt}".strip()


class Profiles:
    def __init__(self, directory):
        self.dir = Path(directory)

    def path(self, name):
        if not valid_name(name):
            raise ProfileError("nom de profil invalide")
        p = (self.dir / f"{name}.json").resolve()
        if self.dir.resolve() not in p.parents:             # défense en profondeur contre un lien symbolique ou un nom détourné
            raise ProfileError("profil hors du dossier autorisé")
        return p

    def names(self):
        try:
            return sorted(p.stem for p in self.dir.glob("*.json") if valid_name(p.stem))
        except OSError:
            return []

    def load(self, name):
        """Profil validé : dict {name, description, system_message, llm, voice, mcp}. Lève ProfileError s'il est absent ou invalide."""
        path = self.path(name)
        try:
            raw = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            raise ProfileError(f"profil « {name} » introuvable")
        except (OSError, ValueError) as e:
            raise ProfileError(f"profil « {name} » illisible : {type(e).__name__}")
        if not isinstance(raw, dict):
            raise ProfileError(f"profil « {name} » : un objet JSON est attendu")
        mcp = raw.get("mcp") or {}
        servers = mcp.get("servers", {}) if isinstance(mcp, dict) else None
        if not isinstance(servers, dict):
            raise ProfileError(f"profil « {name} » : mcp.servers doit être un objet")
        for sname, conf in servers.items():
            if not isinstance(conf, dict) or not (conf.get("url") or conf.get("args") or conf.get("run")):
                raise ProfileError(f"profil « {name} » : serveur « {sname} » sans url ni commande")
        return {"name": name, "description": str(raw.get("description", ""))[:200], "system_message": str(raw.get("system_message", "")),
                "llm": str(raw.get("llm", "")), "voice": bool(raw.get("voice", True)), "mcp": {"servers": servers}}

    def describe(self, name):
        """Aperçu pour la page d'administration (jamais les valeurs de secrets : url, headers et env ne sont pas renvoyés)."""
        p = self.load(name)
        return {"name": name, "description": p["description"], "llm": p["llm"], "voice": p["voice"],
                "servers": [{"name": n, "type": "http" if c.get("url") else "stdio"} for n, c in p["mcp"]["servers"].items()]}
