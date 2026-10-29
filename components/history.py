"""Historique de la conversation envoyé au LLM : plafond de taille et remise à zéro (section "memory" de la config).

Sans limite, l'historique grossit à chaque échange (et chaque résultat d'outil y reste) : requêtes de plus en plus lentes, puis
« Context size has been exceeded » sur un serveur local. Ce module garde l'historique à une taille raisonnable en retirant les
plus anciens échanges COMPLETS (question, appels d'outils, résultats, réponse) : jamais un appel d'outil sans son résultat, ce
qui casserait les modèles de conversation stricts (Mistral).

Les messages sont des dictionnaires au format du moteur (OpenAI : role/content/tool_calls ; Anthropic : role/content avec
blocs). Une « question utilisateur » est un message de rôle user dont le contenu est du texte (pas un résultat d'outil).
"""
import json
import components.utils as utils

DEFAULTS = {
    "reset_on_close": True,     # « Merci Aria » (clôture de conversation) efface l'historique
    "max_exchanges": 12,        # nombre maximal d'échanges conservés (0 = illimité)
    "max_chars": 24000,         # taille maximale de l'historique en caractères, ~6 000 jetons (0 = illimité)
    "tool_result_max_chars": 2000,   # un résultat d'outil plus long est raccourci une fois la réponse donnée (0 = jamais)
}


def settings(params=None):
    return {**DEFAULTS, **(params or {})}


def _is_question(msg):
    """Message utilisateur contenant du texte (début d'un échange), pas un résultat d'outil."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _size(msgs):
    try:
        return len(json.dumps(msgs, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return sum(len(str(m)) for m in msgs)


def _split_exchanges(msgs, first):
    """Indices de début de chaque échange, à partir de `first` (les messages avant sont le message système)."""
    return [i for i in range(first, len(msgs)) if _is_question(msgs[i])]


def trim(msgs, params, first=0):
    """Retire les plus anciens échanges complets tant que les plafonds sont dépassés.
    À appeler AVANT d'ajouter la nouvelle question : elle compte pour un échange (de la place lui est réservée).
    `first` : nombre de messages de tête à conserver (1 pour le message système du moteur OpenAI). Modifie `msgs` sur place."""
    p = settings(params)
    starts = _split_exchanges(msgs, first)
    removed = 0
    while starts:
        too_many = p["max_exchanges"] and len(starts) + 1 > p["max_exchanges"]
        too_big = p["max_chars"] and _size(msgs[first:]) > p["max_chars"]
        if not (too_many or too_big):
            break
        end = starts[1] if len(starts) > 1 else len(msgs)    # dernier échange restant : jusqu'à la fin
        del msgs[starts[0]:end]
        removed += 1
        starts = _split_exchanges(msgs, first)
    if removed:
        utils.log_info("HISTORY", f"Historique allégé : {removed} ancien(s) échange(s) retiré(s), "
                                  f"{len(starts)} conservé(s), ~{_size(msgs[first:])} caractères")
    return removed


def shorten_tool_results(msgs, params, first=0):
    """Raccourcit les gros résultats d'outils des échanges TERMINÉS. À appeler avant d'ajouter la nouvelle question :
    tout l'historique est alors terminé, et les résultats de l'échange en cours restent intacts (le modèle en a besoin)."""
    limit = settings(params)["tool_result_max_chars"]
    if not limit:
        return 0
    cut = 0
    for m in msgs[first:]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > limit:
            m["content"] = m["content"][:limit] + f"\n[... résultat raccourci : {len(m['content']) - limit} caractères retirés]"
            cut += 1
        elif m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), str) \
                        and len(b["content"]) > limit:
                    b["content"] = b["content"][:limit] + f"\n[... résultat raccourci : {len(b['content']) - limit} caractères retirés]"
                    cut += 1
    if cut:
        utils.log_info("HISTORY", f"{cut} ancien(s) résultat(s) d'outil raccourci(s) à {limit} caractères")
    return cut


def reset(store, user=None):
    """Vide l'historique d'un utilisateur (ou de tous si `user` est None). `store` : le dict user_aware_messages du moteur."""
    if user is None:
        n = len(store)
        store.clear()
    else:
        n = 1 if user in store else 0
        store.pop(user, None)
    utils.log_info("HISTORY", f"🧹 Historique du LLM effacé ({n} conversation(s))")
    return n
