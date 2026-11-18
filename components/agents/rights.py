"""Droits des agents. Bibliothèque standard seulement.

Un droit est une capacité qu'Aria a toujours eue et qu'un agent n'a pas par défaut. L'agent par défaut `aria` (`system: true`) les a tous, verrouillés ; un autre
agent peut en recevoir un ou plusieurs de l'administrateur. Ce module est le SEUL endroit qui interprète les droits : le reste du code appelle `has(rec, nom)`.

  local_llm       moteur de langage local (llama-cpp) autorisé, en plus des LLM distants
  all_mcp         tous les serveurs MCP de la configuration, y compris ceux déclarés privés (au lieu de ceux de son profil)
  global_speaker  parle « comme Aria » : interlocuteur global (liste déroulante), mémoire du foyer, souvenirs partagés, fiches créées sans validation
  no_limits       aucune limite de débit ni de quota quotidien
`publish_links` et `publish_speaker` sont deux autres droits historiques, restés champs de premier niveau d'un agent (`keys.py`) pour ne rien changer au fichier existant.
"""

SYSTEM_ID = "aria"                      # identifiant de l'agent par défaut (non supprimable)
RIGHTS = ("local_llm", "all_mcp", "global_speaker", "no_limits")
LEGACY = ("publish_links", "publish_speaker")
ALL = RIGHTS + LEGACY

LABELS = {
    "local_llm": "LLM local",
    "all_mcp": "Tous les serveurs MCP (privés inclus)",
    "global_speaker": "Interlocuteur global et mémoire d'Aria",
    "no_limits": "Sans limite de débit ni de quota",
    "publish_links": "Publier ses propres liens",
    "publish_speaker": "Publier les informations de son interlocuteur",
}


def clean(value):
    """`rights` tel que stocké : seulement les droits connus, en booléens (un nom inconnu ou une valeur absurde est ignoré, jamais accordé)."""
    if not isinstance(value, dict):
        return {name: False for name in RIGHTS}
    return {name: value.get(name) is True for name in RIGHTS}          # exactement `true` : « oui », 1 ou « true » ne sont PAS un droit accordé


def is_system(rec):
    return bool(rec) and bool(rec.get("system"))


def effective(rec):
    """Tous les droits d'un agent, valeurs EFFECTIVES : un agent système les a tous, les autres ce que l'administrateur a accordé."""
    if not rec:
        return {name: False for name in ALL}
    if is_system(rec):
        return {name: True for name in ALL}
    own = clean(rec.get("rights"))
    own.update({name: bool(rec.get(name, False)) for name in LEGACY})
    return own


def has(rec, name):
    """Cet agent a-t-il ce droit ? Un nom inconnu n'est jamais accordé."""
    if name not in ALL:
        return False
    return effective(rec)[name]
