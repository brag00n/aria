"""Compatibilité avec l'ancien vocabulaire (« accès tiers » -> « agent »). Bibliothèque standard seulement.

Un agent est une entité IA qui dialogue avec un interlocuteur. Ce qui change dans les fichiers d'Aria :
  - section de configuration `access` -> `agents`        (l'ancienne reste LUE tant que `agents` est absente ; l'onglet Paramètres écrit `agents`)
  - dossier de données `models/api_access/` -> `models/agents/`   (renommé au démarrage, une seule fois, sans jamais écraser)
  - fichier `keys.json` (`{"keys": [...]}`) -> `agents.json` (`{"agents": [...]}`)   (lu dans les deux formats ; l'ancien est conservé en `.migre`)
Ce qui NE change PAS : les routes `/v1/…`, le client JavaScript (`AriaClient`, `aria-client.js`), les en-têtes `X-Memory-*`, le préfixe des clés API (`ak_`).
"""
import json
import os
import shutil
from pathlib import Path

NEW_KEY, OLD_KEY = "agents", "access"
NEW_DIR, OLD_DIR = "models/agents", "models/api_access"
NEW_FILE, OLD_FILE = "agents.json", "keys.json"


def section(config):
    """Section de configuration des agents : `agents`, sinon l'ancienne `access`, sinon {}. Toujours un dict (jamais None)."""
    config = config or {}
    for name in (NEW_KEY, OLD_KEY):
        value = config.get(name)
        if isinstance(value, dict):
            return value
    return {}


def section_name(config):
    """Nom sous lequel la section est actuellement stockée dans `config` (pour écrire au même endroit qu'on a lu)."""
    config = config or {}
    if isinstance(config.get(NEW_KEY), dict):
        return NEW_KEY
    return OLD_KEY if isinstance(config.get(OLD_KEY), dict) else NEW_KEY


def write_section(config, value):
    """Écrit la section sous le NOUVEAU nom, et retire l'ancienne (une seule source de vérité). UNE affectation de `config[NEW_KEY]`."""
    config[NEW_KEY] = value
    config.pop(OLD_KEY, None)
    return value


def migrate_config(config):
    """Renomme `access` en `agents` dans la config en mémoire. Retourne True si quelque chose a changé. Si les deux existent, `agents` gagne
    et `access` est laissée intacte (rien n'est perdu : on ne fusionne pas à l'aveugle)."""
    if isinstance(config, dict) and isinstance(config.get(OLD_KEY), dict) and NEW_KEY not in config:
        config[NEW_KEY] = config.pop(OLD_KEY)
        return True
    return False


def data_dir(root, configured=None):
    """Dossier des données des agents, sous `root`. `configured` : valeur de `data_dir` de la config (None = défaut). Si seul l'ancien dossier existe,
    il est renommé en `models/agents` (une fois). Si les deux existent, le nouveau est utilisé et l'ancien n'est PAS touché. Retourne un Path."""
    root = Path(root)
    wanted = configured or NEW_DIR
    new = root.joinpath(*wanted.split("/"))
    if configured and configured != OLD_DIR:
        return new                                         # chemin choisi explicitement par l'administrateur : on le respecte tel quel
    old = root.joinpath(*OLD_DIR.split("/"))
    if configured == OLD_DIR:
        new = root.joinpath(*NEW_DIR.split("/"))           # l'ancienne valeur par défaut écrite dans la config : même migration
    if old.is_dir() and not new.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
        except OSError:
            return old                                     # impossible de renommer (verrou, droits) : on continue avec l'ancien, sans perte
    return new


def read_agents_file(directory):
    """Contenu `{"agents": [...]}` du dossier de données, en lisant au besoin l'ancien format. Retourne (liste, chemin du fichier lu ou None)."""
    directory = Path(directory)
    for name, key in ((NEW_FILE, "agents"), (OLD_FILE, "keys")):
        path = directory / name
        try:
            obj = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return obj[key], path
    return [], None


def retire_old_file(directory):
    """Après une première écriture au nouveau format : l'ancien `keys.json` est conservé sous le nom `keys.json.migre` (jamais supprimé)."""
    old = Path(directory) / OLD_FILE
    if old.exists():
        target = old.with_name(OLD_FILE + ".migre")
        if not target.exists():
            try:
                shutil.move(str(old), str(target))
            except OSError:
                pass
