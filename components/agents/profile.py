"""Profils d'accès : ce qu'une clé API a le droit d'utiliser. Bibliothèque standard seulement.

Les profils sont dans `configs/default.json`, section `access.profiles` (éditée dans l'onglet Paramètres) :
    "agents": {"private_servers": [],
               "profiles": {"portail": {"description": "Portail RH", "system_message": "Tu aides les salariés…", "llm": "Llm_Anthropic",
                                        "voice": true, "servers": ["meteo_local", "recherche_web"]}}}

`servers` est une liste de NOMS de serveurs MCP déjà déclarés dans `mcp.params.servers` : un profil ne redéclare rien et ne copie aucun
secret (`env`, `headers` et leurs tags `<NOM>` restent dans la déclaration du serveur). Les serveurs de `agents.private_servers`
(défaut : aucun) ne peuvent jamais être donnés à un agent. Le serveur de MÉMOIRE d'Aria peut l'être, mais seulement par son canal HTTP
(`MEMORY_HTTP_PORT`) : la session tierce s'y connecte avec l'identité de son accès (une personne système, voir tools/memory) et ne lance jamais
son propre processus. Sans profil, une clé n'a AUCUN outil (jamais ceux d'Aria).
"""
import json
import re
import unicodedata

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,59}$")
# Modules LLM autorisés pour des agents : jamais le moteur local (llama-cpp), qui est lourd, synchrone et sans historique
REMOTE_BACKENDS = frozenset(("anthropic", "llm_studio", "ollama", "openai"))


def valid_name(name):
    return bool(NAME_RE.match(str(name or "")))


class ProfileError(ValueError):
    pass


# Prompt système fourni par l'application (option systemPrompt du client) : il COMPLÈTE celui du profil, jamais ne le remplace
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


def clean_speaker(value):
    """Interlocuteur fourni par l'application : {"name": prénom, "family": nom de famille} -> {"n", "c"} nettoyé (60 caractères, sans caractère de contrôle),
    ou None si absent. Lève AppPromptError(400) si la forme est mauvaise. Ce n'est PAS une authentification : l'application affirme qui parle."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AppPromptError(400, "speaker : un objet {name, family} est attendu")

    def propre(x):
        if x is None:
            return ""
        if not isinstance(x, str):
            raise AppPromptError(400, "speaker : name et family sont des textes")
        return " ".join("".join(c for c in x if unicodedata.category(c) not in ("Cc", "Cs")).split())[:60]
    name, family = propre(value.get("name")), propre(value.get("family"))
    if not name:
        raise AppPromptError(400, "speaker : name (prénom) est obligatoire")
    return {"n": name, "c": family}


def compose_system(base, app_prompt):
    """Message système d'une session : celui du profil (ou du module LLM), puis les consignes de l'application sous un titre."""
    base = (base or "").strip()
    if not app_prompt:
        return base
    return f"{base}\n\n{APP_HEADER}\n{app_prompt}".strip()


from . import compat  # noqa: E402

PROFILE_KEYS = frozenset(("description", "system_message", "llm", "voice", "servers", "conversation"))
DEFAULT_PRIVATE = ()


def private_servers(config):
    """Serveurs MCP qu'aucun profil ne peut exposer : `agents.private_servers` (aucun par défaut)."""
    access = compat.section(config)
    value = access.get("private_servers", DEFAULT_PRIVATE)
    return set(value) if isinstance(value, (list, tuple)) else set(DEFAULT_PRIVATE)


def declared_servers(config):
    """Serveurs MCP déclarés dans la config d'Aria : {nom: configuration}."""
    servers = (((config or {}).get("mcp") or {}).get("params") or {}).get("servers")
    return servers if isinstance(servers, dict) else {}


def hosted_only(config):
    """Serveurs démarrés par Aria pour les agents seulement (`agents.hosted_only`) : mémoire d'un agent hors foyer, avec son dossier disjoint."""
    access = compat.section(config)
    value = access.get("hosted_only", [])
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def validate_hosted(config):
    """Erreurs de `agents.hosted_only` : serveurs déclarés, dotés d'un canal HTTP, sur un dossier de données et un port DISTINCTS de ceux des autres
    serveurs de mémoire (deux processus sur le même dossier se verrouillent mutuellement)."""
    errors = []
    declared = declared_servers(config)
    value = compat.section(config).get("hosted_only", [])
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return ["agents.hosted_only doit être une liste de noms de serveurs MCP"]
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


CONVERSATION_KEYS = frozenset(("enabled", "open_phrases", "close_phrases", "stop_phrases"))
MAX_PHRASES = 40
MAX_PHRASE = 60


def validate_conversation(value):
    """Erreurs de l'option `conversation` d'un profil : {enabled, open_phrases, close_phrases, stop_phrases}. Absente = désactivée (comportement d'avant)."""
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["conversation doit être un objet {enabled, open_phrases, close_phrases, stop_phrases}"]
    errors = [f"conversation : clé inconnue « {k} » (attendu : {', '.join(sorted(CONVERSATION_KEYS))})" for k in value if k not in CONVERSATION_KEYS]
    if "enabled" in value and not isinstance(value["enabled"], bool):
        errors.append("conversation.enabled doit être true ou false")
    for k in ("open_phrases", "close_phrases", "stop_phrases"):
        v = value.get(k)
        if k in value and not (isinstance(v, list) and len(v) <= MAX_PHRASES and all(isinstance(x, str) and 0 < len(x.strip()) <= MAX_PHRASE for x in v)):
            errors.append(f"conversation.{k} doit être une liste d'au plus {MAX_PHRASES} textes non vides de {MAX_PHRASE} caractères au plus")
    return errors


def validate_profiles(profiles, config, allow_local=False, allow_private=False):
    """Contrôle un dictionnaire de profils {nom: profil} contre la config. Retourne la liste des erreurs (vide = valide), en texte lisible.
    Contrôle : noms de profil, types, clés inconnues (faute de frappe), serveurs inconnus ou privés, LLM distant.
    `allow_local` / `allow_private` : appelés par le code serveur pour un agent qui a les droits « LLM local » / « tous les serveurs MCP » (jamais depuis l'éditeur
    de profils des Paramètres : un profil seul n'accorde aucun droit, il faut que l'AGENT l'ait)."""
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
            elif (module.get("params") or {}).get("backend_type", "local") not in REMOTE_BACKENDS and not allow_local:
                errors.append(f"profil « {name} » : LLM « {llm} » local, seuls les LLM distants sont autorisés pour un agent")
        errors.extend(f"profil « {name} » : {e}" for e in validate_conversation(prof.get("conversation")))
        servers = prof.get("servers", [])
        if not isinstance(servers, list) or not all(isinstance(x, str) for x in servers):
            errors.append(f"profil « {name} » : servers doit être une liste de noms de serveurs MCP")
            continue
        if len(set(servers)) != len(servers):
            errors.append(f"profil « {name} » : serveur en double dans servers")
        for srv in servers:
            if srv in private and not allow_private:
                errors.append(f"profil « {name} » : le serveur « {srv} » est privé et ne peut pas être donné à un agent")
            elif srv not in declared:
                errors.append(f"profil « {name} » : serveur « {srv} » inconnu (déclarés : {', '.join(sorted(declared)) or 'aucun'})")
            elif _is_memory(declared[srv]) and not _has_http(declared[srv]):
                errors.append(f"profil « {name} » : le serveur de mémoire « {srv} » n'a pas de canal HTTP (MEMORY_HTTP_PORT) : un agent ne peut "
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
    """Valide le JSON des profils puis l'applique par UNE SEULE affectation de `config["agents"]["profiles"]` (les autres clés de `agents`
    sont conservées ; le thread de l'API, qui relit la config vivante, voit l'ancien ou le nouvel ensemble, jamais un état intermédiaire).
    Rien n'est modifié en cas d'erreur. Retourne (ok, message d'erreur, noms des profils modifiés, nouveaux profils)."""
    new, err = parse_profiles(text)
    if err:
        return False, err, [], None
    errors = validate_profiles(new, config)
    if errors:
        return False, "profils d'accès refusés : " + " ; ".join(errors), [], None
    access = compat.section(config)
    changed = changed_profiles(access.get("profiles"), new)
    compat.write_section(config, {**access, "profiles": new})
    return True, "", changed, new


class ConfigProfiles:
    """Profils lus dans la config VIVANTE (`get_config()` à chaque appel) : une modification faite dans l'onglet Paramètres vaut pour les
    sessions créées ensuite, sans redémarrage. Même interface que l'ancienne lecture de fichiers : names, load, describe."""

    def __init__(self, get_config):
        self.get_config = get_config

    def _profiles(self):
        access = compat.section(self.get_config())
        profiles = access.get("profiles")
        return profiles if isinstance(profiles, dict) else {}

    def names(self):
        return sorted(n for n in self._profiles() if valid_name(n))

    def load(self, name, allow_local=False, allow_private=False, all_servers=False):
        """Profil résolu : {name, description, system_message, llm, voice, mcp: {servers: {nom: configuration}}}. Lève ProfileError.
        Les trois options viennent des DROITS de l'agent (rights.py), jamais du profil : LLM local autorisé, serveurs privés autorisés dans la liste du profil,
        et `all_servers` = tous les serveurs MCP déclarés (droit « tous les serveurs MCP »)."""
        if not valid_name(name):
            raise ProfileError("nom de profil invalide")
        config = self.get_config() or {}
        raw = self._profiles().get(name)
        if raw is None:
            raise ProfileError(f"profil « {name} » introuvable")
        errors = validate_profiles({name: raw}, config, allow_local=allow_local, allow_private=allow_private or all_servers)
        if errors:
            raise ProfileError(errors[0])
        declared = declared_servers(config)
        if all_servers:
            servers = dict(declared)                          # tous les serveurs de la config, y compris privés
        else:
            servers = {n: declared[n] for n in raw.get("servers", [])}
        return {"name": name, "description": str(raw.get("description", ""))[:200], "system_message": str(raw.get("system_message", "")),
                "llm": str(raw.get("llm", "")), "voice": bool(raw.get("voice", True)), "mcp": {"servers": servers},
                "conversation": raw.get("conversation") if isinstance(raw.get("conversation"), dict) else None}

    def describe(self, name):
        """Aperçu pour la page d'administration : jamais de secrets (url, headers et env ne sont pas renvoyés), seulement nom et type."""
        p = self.load(name)
        return {"name": name, "description": p["description"], "llm": p["llm"], "voice": p["voice"],
                "servers": [{"name": n, "type": "http" if c.get("url") else "stdio"} for n, c in p["mcp"]["servers"].items()]}
