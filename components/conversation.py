"""Sessions de conversation vocale et arrêt d'Aria (réglages : section "global" -> "conversation" de la config).

Une session s'ouvre quand une phrase DITE commence par un mot d'ouverture (« Bonjour Aria », suivi ou non d'une question), et se ferme après
une phrase contenant un mot de clôture (« Merci Aria », « Bonne nuit »). Entre les deux, on échange librement.
Hors session, les phrases dites sont ignorées en silence (ni LLM, ni message dans le chat).
Le texte saisi au clavier part toujours : il ne peut pas ouvrir de session, mais un mot de clôture la ferme.
Les mots d'arrêt (« stop », « tais-toi »...) coupent Aria à tout moment, voix comme texte.

Module isolé : app.py n'a qu'à instancier Conversation, appeler classify() et verrouiller avec lock.
"""
import asyncio
import re
import unicodedata
import components.utils as utils

DEFAULTS = {
    "enabled": False,    # False : pas de mot d'ouverture ni de clôture, Aria répond toujours (les mots d'arrêt restent actifs)
    "open_phrases": ["bonjour aria", "salut aria", "dis moi aria", "hey aria", "ok aria", "aria"],
    "close_phrases": ["merci aria", "bonne nuit", "cloturons le sujet", "au revoir aria", "a bientot aria"],
    "stop_phrases": ["stop", "tais toi", "silence", "arrete", "ca suffit", "chut"],
}


def normalize_text(text):
    """Minuscules, sans accents ni ponctuation : « Merci, Aria ! » -> « merci aria »."""
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class Conversation:
    def __init__(self, params=None):
        p = {**DEFAULTS, **(params or {})}
        self.enabled = bool(p["enabled"])
        self.open = [normalize_text(x) for x in p["open_phrases"] if normalize_text(x)]
        self.close = [normalize_text(x) for x in p["close_phrases"] if normalize_text(x)]
        self.stop = [normalize_text(x) for x in p["stop_phrases"] if normalize_text(x)]
        self.active = False
        self.lock = asyncio.Lock()        # une seule réponse à la fois ; l'arrêt, lui, ne l'attend pas
        self.stop_requested = False
        utils.log_info("CONV", f"Sessions de conversation : {'activées' if self.enabled else 'désactivées'} | "
                               f"{len(self.open)} mot(s) d'ouverture, {len(self.close)} de clôture, {len(self.stop)} d'arrêt")

    @staticmethod
    def _has(norm, phrases):
        padded = f" {norm} "
        return any(f" {p} " in padded for p in phrases)

    @staticmethod
    def _starts(norm, phrases):
        """L'ouverture doit COMMENCER la phrase (mots entiers), la suite est libre :
        « dis moi aria quelle est la capitale de la france » ouvre la conversation et pose la question."""
        padded = f"{norm} "
        return any(padded.startswith(f"{p} ") for p in phrases)

    def _is_stop(self, norm):
        # Phrase courte seulement : « stop » ou « Aria, tais-toi », pas « il ne faut pas s'arrêter de... »
        words = len(norm.split())
        return any(f" {p} " in f" {norm} " and words <= len(p.split()) + 2 for p in self.stop)

    def classify(self, text, from_voice):
        """Que faire de cette phrase ? stop | ignore | open | close | say"""
        norm = normalize_text(text)
        if not norm:
            return "ignore"
        if self._is_stop(norm):
            return "stop"
        if not self.enabled:
            return "say"
        if self.active:
            return "close" if self._has(norm, self.close) else "say"
        if not from_voice:
            return "say"
        return "open" if self._starts(norm, self.open) else "ignore"

    def begin(self):
        self.active = True
        utils.log_info("CONV", "💬 Conversation ouverte")

    def end(self):
        self.active = False
        utils.log_info("CONV", "💤 Conversation fermée")

    def request_stop(self):
        self.stop_requested = True
        utils.log_info("CONV", "⏹️ Arrêt d'Aria demandé")
