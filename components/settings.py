"""Sauvegarde de la configuration depuis l'onglet Paramètres : tout est analysé et validé AVANT la moindre modification, puis appliqué d'un bloc.

Fonctions pures (sans Gradio) : `app.py` leur passe la config en mémoire et les textes JSON des champs. Une erreur de validation ne laisse jamais
`config` à moitié modifié, ni le fichier à moitié écrit (écriture dans un fichier temporaire puis remplacement atomique).
"""
import json
import os
import tempfile

from components.access import profile as access_profile
from components.language import normalize


def _object(text, label):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"❌ La section {label} doit être un objet JSON")
    return value


def parse_settings(config, fields):
    """Analyse et valide tous les champs sans toucher à `config`. `fields` : dict de textes JSON (global, vad, conversation, memory, stt, llm,
    tts, mcp, profiles) et de choix de modules (vad_choice, stt_choice, llm_choice, tts_choice). Retourne le dict des valeurs analysées ;
    lève ValueError (message prêt à afficher) ou json.JSONDecodeError."""
    out = {}
    out["global"] = _object(fields["global"], "globale")
    lang = out["global"].get("language")
    if normalize(lang) is None:
        raise ValueError(f"❌ Langue '{lang}' invalide : un code est attendu (ex. fr, en, es)")
    if "default_language" in out["global"] and normalize(out["global"]["default_language"]) is None:
        raise ValueError(f"❌ Langue de repli '{out['global']['default_language']}' invalide : un code est attendu (ex. en)")
    out["global"]["language"] = normalize(lang)
    if "default_language" in out["global"]:
        out["global"]["default_language"] = normalize(out["global"]["default_language"])
    out["conversation"] = _object(fields["conversation"], "conversation")
    for key in ("open_phrases", "close_phrases", "stop_phrases"):
        v = out["conversation"].get(key)
        if key in out["conversation"] and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            raise ValueError(f"❌ conversation.{key} doit être une liste de textes")
    out["memory"] = _object(fields["memory"], "mémoire")
    for key in ("max_exchanges", "max_chars", "tool_result_max_chars"):
        v = out["memory"].get(key)
        if key in out["memory"] and not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
            raise ValueError(f"❌ memory.{key} doit être un entier positif ou nul")
    for cat in ("vad", "stt", "llm", "tts"):
        choice = fields[f"{cat}_choice"]
        if choice not in (config.get(cat) or {}):
            raise ValueError(f"❌ Module {cat} inconnu : {choice}")
        out[cat] = _object(fields[cat], f"{cat.upper()}")
    out["mcp"] = _object(fields["mcp"], "MCP")
    # Profils d'accès tiers : validés contre les serveurs MCP et les LLM de la config ENREGISTRÉE avec cette même sauvegarde
    futur = {**config, "mcp": {**(config.get("mcp") or {}), "params": out["mcp"]},
             "llm": {**(config.get("llm") or {}), fields["llm_choice"]: {**(config["llm"].get(fields["llm_choice"]) or {}), "params": out["llm"]}},
             "access": config.get("access") if isinstance(config.get("access"), dict) else {}}
    profiles, err = access_profile.parse_profiles(fields.get("profiles", ""))
    if err:
        raise ValueError("❌ " + err[0].upper() + err[1:])
    errors = access_profile.validate_profiles(profiles, futur) + access_profile.validate_hosted(futur)
    if errors:
        raise ValueError("❌ Profils d'accès refusés : " + " ; ".join(errors))
    out["profiles"] = profiles
    out["audit"] = parse_audit(fields.get("audit"))
    return out


def parse_audit(text):
    """Section « audit » (journal des échanges) : activer, durée de conservation par défaut (jours), texte complet ou empreinte seulement, dossier.
    Texte vide = section non modifiée (None). Valide le type de chaque clé : une valeur absurde ne doit jamais désactiver la purge par erreur."""
    if text is None or not str(text).strip():
        return None
    audit = _object(text, "audit")
    inconnues = sorted(set(audit) - {"enabled", "retention_days", "log_text", "dir"})
    if inconnues:
        raise ValueError("❌ audit : clé(s) inconnue(s) : " + ", ".join(inconnues))
    for key in ("enabled", "log_text"):
        if key in audit and not isinstance(audit[key], bool):
            raise ValueError(f"❌ audit.{key} doit être true ou false")
    v = audit.get("retention_days")
    if "retention_days" in audit and not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 3650):
        raise ValueError("❌ audit.retention_days doit être un nombre de jours entier entre 1 et 3650")
    d = audit.get("dir")
    if "dir" in audit and not (isinstance(d, str) and d.strip() and not os.path.isabs(d) and ".." not in d.replace("\\", "/").split("/")):
        raise ValueError("❌ audit.dir doit être un chemin relatif au dossier d'Aria, sans « .. »")
    return audit


def apply_settings(config, fields):
    """Valide (parse_settings) puis applique à `config` d'un bloc. Retourne (parsed, changed) : `parsed` les valeurs appliquées et `changed` les
    noms de profils d'accès ajoutés, retirés ou modifiés. Lève ValueError / json.JSONDecodeError sans avoir rien modifié."""
    parsed = parse_settings(config, fields)
    access = config.get("access") if isinstance(config.get("access"), dict) else {}
    changed = access_profile.changed_profiles(access.get("profiles"), parsed["profiles"])
    config["global"] = parsed["global"]
    config["conversation"] = parsed["conversation"]
    config["memory"] = parsed["memory"]
    config["active_modules"] = {cat: fields[f"{cat}_choice"] for cat in ("vad", "stt", "llm", "tts")}
    for cat in ("vad", "stt", "llm", "tts"):
        config[cat][fields[f"{cat}_choice"]]["params"] = parsed[cat]
    config["mcp"]["params"] = parsed["mcp"]
    if parsed["audit"] is not None:
        config["audit"] = parsed["audit"]
    config["access"] = {**access, "profiles": parsed["profiles"]}        # UNE affectation : le thread de l'API ne voit jamais un état intermédiaire
    return parsed, changed


def write_config(config, path):
    """Écrit la config en UTF-8 (les accents restent lisibles) dans un fichier temporaire puis le substitue au fichier réel : un échec en cours
    d'écriture ne laisse jamais un fichier tronqué."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def profiles_help(config):
    """Aide affichée sous l'éditeur de profils : serveurs MCP utilisables et serveurs privés exclus."""
    exposable = access_profile.exposable_servers(config)
    private = sorted(access_profile.private_servers(config) & set(access_profile.declared_servers(config)))
    return ("<small>Un profil choisit ce qu'une clé API d'application tierce peut utiliser. Clés d'un profil : "
            "<code>description</code>, <code>system_message</code>, <code>llm</code> (module LLM distant, vide = module actif), "
            "<code>voice</code> et <code>servers</code> (noms de serveurs MCP). "
            f"<br>Serveurs utilisables : <b>{', '.join(exposable) or 'aucun'}</b>"
            + (f" · jamais exposés : {', '.join(private)}" if private else "")
            + ". Les profils sont appliqués sans redémarrage aux nouvelles conversations.</small>")


def save_message(changed, closed, warnings):
    """Message de fin de sauvegarde."""
    parts = ["✅ Configuration sauvegardée ! Cliquez sur « Redémarrer » pour appliquer les changements des autres sections."]
    if changed:
        parts.append(f"Profils d'accès appliqués sans redémarrage : {', '.join(changed)}.")
    if closed:
        parts.append(f"{closed} conversation(s) ouverte(s) fermée(s).")
    parts.extend("⚠️ " + w for w in warnings)
    return "\n\n".join(parts)
