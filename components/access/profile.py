"""Profils d'accès : ce qu'une clé API a le droit d'utiliser. Bibliothèque standard seulement.

Les profils sont dans `configs/default.json`, section `access.profiles` (éditée dans l'onglet Paramètres) :
    "access": {"private_servers": [],
               "profiles": {"portail": {"description": "Portail RH", "system_message": "Tu aides les salariés…", "llm": "Llm_Anthropic",
                                        "voice": true, "servers": ["meteo_local", "recherche_web"]}}}

`servers` est une liste de NOMS de serveurs MCP déjà déclarés dans `mcp.params.servers` : un profil ne redéclare rien et ne copie aucun
secret (`env`, `headers` et leurs tags `<NOM>` restent dans la déclaration du serveur). Les serveurs de `access.private_servers`
(défaut : aucun) ne peuvent jamais être donnés à un tiers. Le serveur de MÉMOIRE d'Aria peut l'être, mais seulement par son canal HTTP
(`MEMORY_HTTP_PORT`) : la session tierce s'y connecte avec l'identité de son accès (une personne système, voir tools/memory) et ne lance jamais
son propre processus. Sans profil, une clé n'a AUCUN outil (jamais ceux d'Aria).
"""
import json
import re
import unicodedata

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


PROFILE_KEYS = frozenset(("description", "system_message", "llm", "voice", "servers"))
DEFAULT_PRIVATE = ()


def private_servers(config):
    """Serveurs MCP qu'aucun profil ne peut exposer : `access.private_servers` (aucun par défaut)."""
    access = (config or {}).get("access") or {}
    value = access.get("private_servers", DEFAULT_PRIVATE)
    return set(value) if isinstance(value, (list, tuple)) else set(DEFAULT_PRIVATE)


def declared_servers(config):
    """Serveurs MCP déclarés dans la config d'Aria : {nom: configuration}."""
    servers = (((config or {}).get("mcp") or {}).get("params") or {}).get("servers")
    return servers if isinstance(servers, dict) else {}


def hosted_only(config):
    """Serveurs démarrés par Aria pour les accès tiers seulement (`access.hosted_only`) : mémoire d'un tiers hors foyer, avec son dossier disjoint."""
    access = (config or {}).get("access") or {}
    value = access.get("hosted_only", [])
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def validate_hosted(config):
    """Erreurs de `access.hosted_only` : serveurs déclarés, dotés d'un canal HTTP, sur un dossier de données et un port DISTINCTS de ceux des autres
    serveurs de mémoire (deux processus sur le même dossier se verrouillent mutuellement)."""
    errors = []
    declared = declared_servers(config)
    value = ((config or {}).get("access") or {}).get("hosted_only", [])
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return ["access.hosted_only doit être une liste de noms de serveurs MCP"]
    memoires = {n: c for n, c in declared.items() if _is_memory(c)}
    dossiers, ports = {}, {}
    for n, c in memoires.items():
        env = c.get("env") or {}
        dossiers.setdefault(str(env.get("MEMORY_DIR", "")) + "/" + str(env.get("MEMORY_NAMESPACE", "default")), []).append(n)
        if _has_http(c):
            from .memory_link import http_port
            ports.setdefault(http_port(c), []).append(n)
    for d, noms in dossiers.items():
        if d.strip("/") and len(noms) > 1:
            errors.append(f"les serveurs de mémoire {', '.join(sorted(noms))} partagent le même dossier de données ({d.strip('/')}) : ils se verrouilleraient")
    for p, noms in ports.items():
        if len(noms) > 1:
            errors.append(f"les serveurs de mémoire {', '.join(sorted(noms))} partagent le même port HTTP ({p})")
    for n in value:
        if n not in declared:
            errors.append(f"hosted_only : serveur « {n} » inconnu")
        elif not _has_http(declared[n]):
            errors.append(f"hosted_only : le serveur « {n} » n'a pas de canal HTTP (MEMORY_HTTP_PORT)")
        elif "MEMORY_UI_PORT" in (declared[n].get("env") or {}):
            errors.append(f"hosted_only : le serveur « {n} » déclare une page de gestion (MEMORY_UI_PORT) : un seul serveur la porte")
    return errors


def exposable_servers(config):
    """Noms des serveurs qu'un profil peut référencer : déclarés, non privés, et — pour la mémoire — dotés d'un canal HTTP (voir validate_profiles)."""
    private, declared = private_servers(config), declared_servers(config)
    return sorted(n for n, conf in declared.items() if n not in private and not (_is_memory(conf) and not _has_http(conf)))


def _is_memory(conf):
    """Serveur de mémoire d'Aria : il lance tools/memory/server.py (ou déclare un dossier de données)."""
    if not isinstance(conf, dict):
        return False
    texte = " ".join(str(x) for x in (conf.get("args") or [])) + " " + str(conf.get("run", ""))
    env = conf.get("env") or {}
    return "tools/memory/server.py" in texte.replace("\\", "/") or "MEMORY_DIR" in env or "MEMORY_HTTP_PORT" in env


def _has_http(conf):
    from .memory_link import http_port
    return http_port(conf) is not None


def validate_profiles(profiles, config):
    """Contrôle un dictionnaire de profils {nom: profil} contre la config. Retourne la liste des erreurs (vide = valide), en texte lisible.
    Contrôle : noms de profil, types, clés inconnues (faute de frappe), serveurs inconnus ou privés, LLM distant."""
    errors = []
    if not isinstance(profiles, dict):
        return ["les profils doivent former un objet JSON {nom: profil}"]
    declared, private = declared_servers(config), private_servers(config)
    llm_conf = (config or {}).get("llm") or {}
    for name, prof in profiles.items():
        if not valid_name(name):
            errors.append(f"profil « {name} » : nom invalide (lettres, chiffres, - et _, 60 caractères au plus)")
            continue
        if not isinstance(prof, dict):
            errors.append(f"profil « {name} » : un objet est attendu")
            continue
        for k in prof:
            if k not in PROFILE_KEYS:
                errors.append(f"profil « {name} » : clé inconnue « {k} » (attendu : {', '.join(sorted(PROFILE_KEYS))})")
        for k in ("description", "system_message", "llm"):
            if k in prof and not isinstance(prof[k], str):
                errors.append(f"profil « {name} » : {k} doit être un texte")
        if "voice" in prof and not isinstance(prof["voice"], bool):
            errors.append(f"profil « {name} » : voice doit être true ou false")
        llm = prof.get("llm")
        if isinstance(llm, str) and llm:
            module = llm_conf.get(llm)
            if not isinstance(module, dict):
                errors.append(f"profil « {name} » : LLM « {llm} » inconnu")
            elif (module.get("params") or {}).get("backend_type", "local") not in REMOTE_BACKENDS:
                errors.append(f"profil « {name} » : LLM « {llm} » local, seuls les LLM distants sont autorisés pour un tiers")
        servers = prof.get("servers", [])
        if not isinstance(servers, list) or not all(isinstance(x, str) for x in servers):
            errors.append(f"profil « {name} » : servers doit être une liste de noms de serveurs MCP")
            continue
        if len(set(servers)) != len(servers):
            errors.append(f"profil « {name} » : serveur en double dans servers")
        for srv in servers:
            if srv in private:
                errors.append(f"profil « {name} » : le serveur « {srv} » est privé et ne peut pas être donné à un tiers")
            elif srv not in declared:
                errors.append(f"profil « {name} » : serveur « {srv} » inconnu (déclarés : {', '.join(sorted(declared)) or 'aucun'})")
            elif _is_memory(declared[srv]) and not _has_http(declared[srv]):
                errors.append(f"profil « {name} » : le serveur de mémoire « {srv} » n'a pas de canal HTTP (MEMORY_HTTP_PORT) : un tiers ne peut "
                              "pas lancer son propre processus de mémoire")
    return errors


def parse_profiles(text):
    """JSON de l'éditeur -> (dict, None) ou (None, message d'erreur). Un texte vide équivaut à aucun profil."""
    text = (text or "").strip()
    if not text:
        return {}, None
    try:
        value = json.loads(text)
    except ValueError as e:
        return None, f"JSON invalide dans les profils d'accès : {e}"
    if not isinstance(value, dict):
        return None, "les profils doivent former un objet JSON {nom: profil}"
    return value, None


def changed_profiles(old, new):
    """Noms des profils ajoutés, retirés ou modifiés entre deux dictionnaires de profils."""
    old, new = old if isinstance(old, dict) else {}, new if isinstance(new, dict) else {}
    return sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))


def apply_profiles(config, text):
    """Valide le JSON des profils puis l'applique par UNE SEULE affectation de `config["access"]["profiles"]` (les autres clés de `access`
    sont conservées ; le thread de l'API, qui relit la config vivante, voit l'ancien ou le nouvel ensemble, jamais un état intermédiaire).
    Rien n'est modifié en cas d'erreur. Retourne (ok, message d'erreur, noms des profils modifiés, nouveaux profils)."""
    new, err = parse_profiles(text)
    if err:
        return False, err, [], None
    errors = validate_profiles(new, config)
    if errors:
        return False, "profils d'accès refusés : " + " ; ".join(errors), [], None
    access = config.get("access")
    access = access if isinstance(access, dict) else {}
    changed = changed_profiles(access.get("profiles"), new)
    config["access"] = {**access, "profiles": new}
    return True, "", changed, new


class ConfigProfiles:
    """Profils lus dans la config VIVANTE (`get_config()` à chaque appel) : une modification faite dans l'onglet Paramètres vaut pour les
    sessions créées ensuite, sans redémarrage. Même interface que l'ancienne lecture de fichiers : names, load, describe."""

    def __init__(self, get_config):
        self.get_config = get_config

    def _profiles(self):
        access = (self.get_config() or {}).get("access") or {}
        profiles = access.get("profiles")
        return profiles if isinstance(profiles, dict) else {}

    def names(self):
        return sorted(n for n in self._profiles() if valid_name(n))

    def load(self, name):
        """Profil résolu : {name, description, system_message, llm, voice, mcp: {servers: {nom: configuration}}}. Lève ProfileError."""
        if not valid_name(name):
            raise ProfileError("nom de profil invalide")
        config = self.get_config() or {}
        raw = self._profiles().get(name)
        if raw is None:
            raise ProfileError(f"profil « {name} » introuvable")
        errors = validate_profiles({name: raw}, config)
        if errors:
            raise ProfileError(errors[0])
        declared = declared_servers(config)
        servers = {n: declared[n] for n in raw.get("servers", [])}
        return {"name": name, "description": str(raw.get("description", ""))[:200], "system_message": str(raw.get("system_message", "")),
                "llm": str(raw.get("llm", "")), "voice": bool(raw.get("voice", True)), "mcp": {"servers": servers}}

    def describe(self, name):
        """Aperçu pour la page d'administration : jamais de secrets (url, headers et env ne sont pas renvoyés), seulement nom et type."""
        p = self.load(name)
        return {"name": name, "description": p["description"], "llm": p["llm"], "voice": p["voice"],
                "servers": [{"name": n, "type": "http" if c.get("url") else "stdio"} for n, c in p["mcp"]["servers"].items()]}
