"""Langue globale d'Aria : un code (ex. "fr") pour tous les modules, avec repli tracé dans les logs.

Réglages (section "global" de configs/default.json, à côté de "active_modules") :
  language         : code de la langue de toute la chaîne (STT, LLM, TTS), ex. "fr"
  default_language : langue de repli du STT si la langue demandée n'est pas gérée, ex. "en"

Règles de repli :
  VAD  : indépendant de la langue (il détecte la parole, pas les mots)
  STT  : langue non gérée par Whisper -> default_language -> sinon détection automatique
  LLM  : toujours possible, la langue est ajoutée à la consigne système
  TTS  : voix Piper (Sherpa) de la liste PIPER_MODELS -> sinon Kokoro -> sinon aucune voix (réponse écrite seulement)
"""
import importlib.util
import re
from components.utils import log_info

# Modèles Piper connus de sherpa-onnx (release "tts-models"), un par langue. Chaque nom a été vérifié sur la release.
# Seuls le français et l'anglais ont été chargés et testés dans Aria.
PIPER_MODELS = {
    "fr": "vits-piper-fr_FR-upmc-medium",
    "en": "vits-piper-en_US-lessac-medium",
    "es": "vits-piper-es_ES-davefx-medium",
    "de": "vits-piper-de_DE-thorsten-medium",
    "it": "vits-piper-it_IT-paola-medium",
    "pt": "vits-piper-pt_BR-faber-medium",
    "hi": "vits-piper-hi_IN-pratham-medium",
    "zh": "vits-piper-zh_CN-huayan-medium",
}

# Kokoro : code de langue du paquet -> voix par défaut ; les paquets de phonétisation requis pour ja/zh
KOKORO_CODES = {"en": "a", "es": "e", "fr": "f", "hi": "h", "it": "i", "pt": "p", "ja": "j", "zh": "z"}
KOKORO_DEFAULT_VOICE = {"a": "af_heart", "b": "bf_emma", "e": "ef_dora", "f": "ff_siwis", "h": "hf_alpha",
                        "i": "if_sara", "p": "pf_dora", "j": "jf_alpha", "z": "zf_xiaoxiao"}
KOKORO_REQUIRES = {"j": ["pyopenjtalk"], "z": ["jieba", "pypinyin"]}

# Noms utilisés dans la consigne donnée au LLM
NAMES = {"fr": "français", "en": "anglais", "es": "espagnol", "de": "allemand", "it": "italien",
         "pt": "portugais", "hi": "hindi", "ja": "japonais", "zh": "chinois", "nl": "néerlandais",
         "ru": "russe", "ar": "arabe", "pl": "polonais", "tr": "turc"}

_CODE = re.compile(r"^[a-z]{2,3}$")
_whisper_cache = []          # [None] = liste illisible ; [set] = codes lus dans la bibliothèque


def normalize(value):
    """'fr', 'FR', 'fr-FR', 'fr_FR' -> 'fr'. Renvoie None si ce n'est pas un code de langue."""
    if not isinstance(value, str):
        return None
    code = re.split(r"[-_]", value.strip().lower())[0]
    return code if _CODE.match(code) else None


def whisper_languages():
    """Codes de langue gérés par Whisper, lus dans la bibliothèque installée (None si illisible)."""
    if not _whisper_cache:
        codes = None
        try:
            from faster_whisper.tokenizer import _LANGUAGE_CODES
            codes = set(_LANGUAGE_CODES)
        except Exception:
            try:
                from transformers.models.whisper.tokenization_whisper import LANGUAGES
                codes = set(LANGUAGES)
            except Exception:
                pass
        _whisper_cache.append(codes)
    return _whisper_cache[0]


def normalize_config(config):
    """Section "global" (réglages généraux) à côté de "active_modules" (modules actifs), tous deux à la racine.
    Migre l'état intermédiaire où active_modules était placé DANS global. Renvoie la section global."""
    glob = config.setdefault("global", {})
    if "active_modules" in glob:
        config.setdefault("active_modules", glob.pop("active_modules"))
    return glob


class Language:
    def __init__(self, code="fr", default="en"):
        self.default = normalize(default) or "en"
        self.code = normalize(code)
        if self.code is None:
            log_info("LANG", f"⚠️ Langue '{code}' invalide (attendu : un code comme 'fr') -> '{self.default}'")
            self.code = self.default

    @classmethod
    def from_config(cls, config):
        glob = normalize_config(config)
        return cls(glob.get("language", "fr"), glob.get("default_language", "en"))

    # --- STT ---------------------------------------------------------------------------------------------
    def stt(self):
        """Langue transmise à Whisper : la langue globale si gérée, sinon la langue de repli, sinon None (auto)."""
        supported = whisper_languages()
        if supported is None:
            log_info("LANG", f"STT : liste des langues de Whisper illisible, '{self.code}' transmis tel quel")
            return self.code
        if self.code in supported:
            log_info("LANG", f"STT : langue '{self.code}'")
            return self.code
        if self.default in supported:
            log_info("LANG", f"STT : '{self.code}' non gérée par Whisper -> repli sur '{self.default}'")
            return self.default
        log_info("LANG", f"STT : '{self.code}' et '{self.default}' non gérées -> détection automatique")
        return None

    # --- LLM ---------------------------------------------------------------------------------------------
    def llm_instruction(self):
        name = NAMES.get(self.code, f"la langue de code « {self.code} »")
        return f"LANGUE : réponds toujours en {name} (code {self.code}), même si l'utilisateur change de langue."

    # --- TTS ---------------------------------------------------------------------------------------------
    def piper_model(self, overrides=None):
        """Nom du modèle Piper pour la langue, ou None s'il n'y en a pas dans la liste connue."""
        model = {**PIPER_MODELS, **(overrides or {})}.get(self.code)
        if model is None:
            log_info("LANG", f"TTS Piper : aucun modèle connu pour '{self.code}'")
        return model

    def kokoro_code(self):
        """Code de langue Kokoro, ou None si la langue n'est pas gérée ou si ses dépendances manquent."""
        code = KOKORO_CODES.get(self.code)
        if code is None:
            log_info("LANG", f"TTS Kokoro : langue '{self.code}' non gérée")
            return None
        missing = [m for m in KOKORO_REQUIRES.get(code, []) if importlib.util.find_spec(m) is None]
        if missing:
            log_info("LANG", f"TTS Kokoro : '{self.code}' demande des paquets absents de l'image ({', '.join(missing)})")
            return None
        return code

    def kokoro_voice(self, code, requested=None):
        """(code Kokoro, voix) : la voix demandée si elle correspond à la langue, sinon la voix par défaut."""
        if requested and len(requested) > 1:
            ok = requested[0] == code or (code == "a" and requested[0] == "b")      # voix anglaises b* = britanniques
            if ok:
                return ("b" if requested[0] == "b" else code), requested
            log_info("LANG", f"TTS Kokoro : voix '{requested}' incompatible avec '{self.code}' -> '{KOKORO_DEFAULT_VOICE[code]}'")
        return code, KOKORO_DEFAULT_VOICE[code]
