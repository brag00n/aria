import asyncio
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import deque

# Racine du projet : ARIA_ROOT si défini, /aria dans Docker, sinon le dossier du projet
ARIA_ROOT = os.environ.get("ARIA_ROOT") or (
    "/aria" if os.path.isdir("/aria")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def aria_path(*parts):
    """Construit un chemin absolu à partir de la racine du projet."""
    return os.path.join(ARIA_ROOT, *parts)

# On garde les 100 dernières lignes en mémoire vive
LOG_BUFFER = deque(maxlen=100)

# --- UTILITAIRE DE LOG ---
def log_info(step, message=""):
    """Affiche un log de performance avec timestamp précis."""
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
    log_entry = f"[{timestamp}] [INFO] | {message} [{step.upper()}]"

    # 1. Affichage Console (standard)
    print(log_entry)

    # 2. Stockage en Mémoire (nouveau)
    LOG_BUFFER.append(log_entry)

def get_logs_from_memory():
    """Récupère les logs formatés pour l'affichage (récents en haut)."""
    # On inverse la liste pour avoir les derniers logs en premier
    return "\n".join(list(LOG_BUFFER))

class StepTimer:
    """Chronomètre par étape : logue le temps depuis la précédente étape et depuis le début."""
    def __init__(self, tag):
        self.tag = tag
        self.t0 = self.last = time.perf_counter()

    def mark(self, step, extra=""):
        now = time.perf_counter()
        log_info(self.tag, f"⏱ {step} : +{now - self.last:.2f}s | cumul {now - self.t0:.2f}s{' | ' + extra if extra else ''}")
        self.last = now

# --- DÉCOUPAGE DU TEXTE EN BLOCS À DIRE ---
SPEECH_STRONG = ".!?\n"        # fin de phrase : coupe toujours
SPEECH_SOFT = ",;:—–"          # pause : coupe si le bloc a déjà assez de caractères
FIRST_CHUNK_MIN = 25           # 1er bloc de la réponse : seuil bas pour démarrer la voix plus tôt
CHUNK_MIN = 40                 # blocs suivants : au moins 40 caractères (évite "Eh bien," seul)
CHUNK_SAFETY_MAX = 200         # filet de sécurité seulement : sans AUCUNE ponctuation au-delà de 200 caractères,
                               # on coupe au dernier espace. Le découpage normal se fait sur la ponctuation ;
                               # c'est la consigne de style donnée au LLM (SPEECH_STYLE_PROMPT) qui garde les phrases courtes.

# Consigne ajoutée au message système de tous les moteurs LLM (voir components/llm.py)
SPEECH_STYLE_PROMPT = (
    "CONSIGNES DE STYLE (tes réponses sont lues à voix haute, phrase par phrase) : "
    "écris des phrases courtes, de 15 mots au maximum, chacune terminée par un point, un point d'exclamation "
    "ou un point d'interrogation. Mets une virgule à chaque pause naturelle. "
    "Évite les incises longues et les énumérations sans virgules. "
    "N'utilise ni liste, ni titre, ni symbole de mise en forme. "
    "Cela limite la longueur des phrases, pas celle de ta réponse."
)
CODE_FENCE = "```"             # délimiteur de bloc de code markdown : son contenu n'est pas dit
ORPHAN_MAX = 20               # un bloc de moins de 20 caractères est fusionné avec le précédent...
MERGE_MAX = 110                # ...si le texte fusionné ne dépasse pas 110 caractères
SPEECH_CLOSERS = "»”’\"')]"    # fermants qui restent collés à la ponctuation qui les précède : `point.»`
SPACED_CLOSERS = "»”"          # fermants acceptés après une espace (typographie française : `point. »`)
                               # " et ' droits sont exclus : après une espace, ils peuvent OUVRIR une citation

def _skip_closers(buf, j, closers=SPEECH_CLOSERS):
    """Indice du premier caractère après les fermants commençant en j."""
    while j < len(buf) and buf[j] in closers:
        j += 1
    return j

def _boundary(buf, i):
    """Frontière après la ponctuation en position i.
    Renvoie l'indice de coupure, None s'il faut attendre la suite du flux, ou -1 s'il n'y a pas de frontière
    ("3,5" : la ponctuation est suivie d'un caractère, pas d'une espace).
    Un fermant `»` peut suivre la ponctuation après des espaces (`jardin. »`) : il reste dans le bloc.
    Si le tampon s'arrête sur la ponctuation ou sur des espaces, on attend le caractère suivant pour savoir
    si un fermant arrive : le résultat ne dépend ainsi pas du rythme d'arrivée du texte.
    Le coût est d'un caractère de plus à attendre, et seulement en fin de tampon."""
    n = len(buf)
    j = _skip_closers(buf, i + 1)
    if j >= n:
        return None                    # fin de tampon : on attend la suite
    if not buf[j].isspace():
        return -1
    k = j
    while k < n and buf[k].isspace():
        k += 1
    if k >= n:
        return None                    # le tampon s'arrête sur des espaces : un fermant peut suivre
    if buf[k] in SPACED_CLOSERS:
        return _skip_closers(buf, k, SPACED_CLOSERS)
    return j

def next_speech_cut(buf, is_first=False):
    """Indice de coupure dans buf (texte à dire = buf[:cut]), ou None s'il faut attendre la suite.
    Une ponctuation ne compte que si elle est suivie d'une espace (après d'éventuels fermants), ce qui évite
    de couper dans "3,5" ou "3.5". Les fermants » " ) restent dans le bloc de la ponctuation qui les précède,
    qu'ils la touchent (`point.»`) ou non (`point. »`).
    Fin de phrase (. ! ?) en fin de tampon : on attend le caractère suivant, car ce peut être un chiffre
    ("0." puis "6018") ou un fermant. Si le flux s'arrête là, l'appelant envoie le reste du tampon au TTS.
    Le premier point de coupure rencontré l'emporte.
    Le seuil des pauses (virgule...) est plus bas pour le premier bloc : FIRST_CHUNK_MIN au lieu de CHUNK_MIN."""
    min_soft = FIRST_CHUNK_MIN if is_first else CHUNK_MIN
    n = len(buf)
    for i, ch in enumerate(buf):
        if ch == "\n":
            return i + 1
        if ch in SPEECH_STRONG:
            b = _boundary(buf, i)
            if b is None:
                return None
            if b >= 0:
                return b
        elif ch in SPEECH_SOFT:
            b = _boundary(buf, i)
            if b is None:
                if i + 1 >= min_soft:
                    return None        # pause assez longue : on attend le caractère suivant avant de couper
            elif b >= min_soft:        # b = -1 (pas de frontière) est toujours < min_soft
                return b
    if n >= CHUNK_SAFETY_MAX:
        sp = buf.rfind(" ")
        if sp > CHUNK_MIN:
            return sp
    return None

# Un seul thread de synthèse : les blocs restent dans l'ordre et le modèle TTS n'est jamais appelé en parallèle
_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

class TtsQueue:
    """Synthèse vocale dans un thread dédié, pendant que le LLM continue de générer.
    Les blocs audio sont restitués dans l'ordre où le texte a été soumis."""
    def __init__(self, tts, user, tag):
        self.tts = tts
        self.user = user
        self.tag = tag
        self.pending = deque()
        self.in_code = False           # dans un bloc ``` : le texte n'est pas lu
        self.count = 0
        self.total = 0.0

    def _run(self, n, text):
        t = time.perf_counter()
        audio = self.tts.run_tts_to_file(text, user_id=self.user)
        dt = time.perf_counter() - t
        self.total += dt
        log_info(self.tag, f"⏱ tts_bloc_{n} : {dt:.2f}s pour {len(text)} caractères")
        return audio

    def _enqueue(self, text):
        self.count += 1
        cf = _TTS_EXECUTOR.submit(self._run, self.count, text)
        # cf (concurrent.futures) sert à annuler un bloc pas encore démarré ; le future asyncio sert à l'attendre
        self.pending.append((text, asyncio.wrap_future(cf, loop=asyncio.get_running_loop()), cf))

    def submit(self, text):
        """Lance la synthèse d'un bloc de texte sans attendre.
        Un bloc orphelin (moins de ORPHAN_MAX caractères) est fusionné avec le précédent, mais seulement si
        celui-ci n'a pas encore démarré sa synthèse : l'annuler ne coûte alors aucun temps. S'il est déjà en
        cours ou terminé, l'orphelin est synthétisé seul (annuler ferait perdre son audio)."""
        if len(text) < ORPHAN_MAX and self.pending:
            prev_text, _, prev_cf = self.pending[-1]
            if len(prev_text) + 1 + len(text) <= MERGE_MAX and prev_cf.cancel():
                self.pending.pop()
                self.count -= 1
                log_info(self.tag, f"⏱ fusion du bloc orphelin ({len(text)} car.) avec le précédent ({len(prev_text)} car.)")
                text = f"{prev_text} {text}"
        self._enqueue(text)

    def flush(self, buf):
        """Fin du flux : envoie le reste du tampon au TTS (fusionné avec le bloc précédent s'il est trop court).
        Un bloc de code resté ouvert est ignoré."""
        buf = self.feed(buf)
        if self.in_code:
            self.in_code = False       # flush est appelé à chaque tour de la boucle d'outils : on repart à zéro
            return
        txt = clean_text_for_tts(remove_nonverbal_cues(buf)).strip()
        if len(txt) > 1:
            self.submit(txt)

    def _speak(self, piece):
        txt = clean_text_for_tts(remove_nonverbal_cues(piece)).strip()
        if len(txt) > 2:
            self.submit(txt)

    def feed(self, buf):
        """Découpe buf en blocs à dire (voir next_speech_cut), les soumet, et renvoie le reste du buffer.
        Le code entre ``` n'est jamais lu : le découpage en phrases coupe un bloc en plusieurs morceaux, donc
        l'état « dans un bloc de code » est suivi ici, d'un appel à l'autre (self.in_code)."""
        while True:
            if self.in_code:
                end = buf.find(CODE_FENCE)
                if end < 0:
                    # On garde les ` de fin : la clôture peut arriver coupée entre deux jetons
                    return buf[len(buf.rstrip("`")):][-2:]
                buf, self.in_code = buf[end + len(CODE_FENCE):], False
                continue
            start = buf.find(CODE_FENCE)
            if start < 0:
                break
            # Le texte avant le code se dit en entier, même sans ponctuation finale
            before, buf, self.in_code = buf[:start], buf[start + len(CODE_FENCE):], True
            while (cut := next_speech_cut(before, is_first=self.count == 0)) is not None:
                self._speak(before[:cut])
                before = before[cut:]
            self._speak(before)
        while (cut := next_speech_cut(buf, is_first=self.count == 0)) is not None:
            self._speak(buf[:cut])
            buf = buf[cut:]
        return buf

    def pop_ready(self):
        """Blocs déjà synthétisés, en tête de file uniquement (ordre conservé)."""
        out = []
        while self.pending and self.pending[0][1].done():
            text, fut, _ = self.pending.popleft()
            out.append((fut.result(), text))
        return out

    async def drain(self):
        """Attend les blocs restants, dans l'ordre."""
        while self.pending:
            text, fut, _ = self.pending.popleft()
            yield await fut, text

# --- CONVERSION CHIFFRES ROMAINS ---
def roman_to_arabic(text):
    roman_map = {
        'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
        'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
        'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
    }
    def parse_roman(m):
        roman = m.group(0)
        i, num = 0, 0
        while i < len(roman):
            if i + 1 < len(roman) and roman[i:i+2] in roman_map:
                num += roman_map[roman[i:i+2]]; i += 2
            else:
                num += roman_map[roman[i]]; i += 1
        return str(num)
    roman_regex = r'\b(?=[MDCLXVI]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
    return re.sub(roman_regex, lambda m: parse_roman(m) if (len(m.group(0))>1 or m.group(0)=='V') else m.group(0), text)

# --- SIGNATURES REQUISES PAR LLM.PY ---
def remove_nonverbal_cues(text):
    """Transforme les indications de type *rit* en "rit"."""
    if not text: return ""
    # Remplace le texte entre astérisques simples par le même texte entre guillemets
    # On utilise une expression régulière qui capture le contenu
    return re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'"\1"', text).strip()

def clean_text_for_tts(text):
    """Nettoyage complet pour la voix avec préservation du ton."""
    if not text: return ""
    
    # 1. SUPPRESSION TOTALE DU CODE
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    
    # 2. TRANSFORMATION DES ACTIONS (*action* -> "action")
    # On le fait AVANT de nettoyer le reste des astérisques
    text = remove_nonverbal_cues(text)
    
    # 3. TRAITEMENT DU GRAS (On enlève juste les symboles)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # 4. Autres nettoyages
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = roman_to_arabic(text)
    text = remove_emojis(text)
    
    # 5. Nettoyage final (On ne touche pas aux guillemets qu'on vient d'ajouter)
    # Astérisques restants (gras coupé entre deux blocs, puces « * ») : sinon la voix dit « astérisque »
    text = text.replace('*', ' ').replace('_', '').replace('`', '')
    return re.sub(r'\s+', ' ', text).strip()

def remove_emojis(text):
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002702-\U000027B0]', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


SECRET_TAG = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")

def expand_secrets(value, where=""):
    """Remplace les tags <NOM> par la variable d'environnement NOM (définie dans .env), dans toute la structure.
    Ex. "Bearer <secretAAA>" avec secretAAA=aaaaa dans .env -> "Bearer aaaaa".
    Un tag sans variable correspondante est laissé tel quel (avertissement dans les logs, sans afficher de valeur).
    La config d'origine n'est pas modifiée : le secret n'existe qu'en mémoire."""
    if isinstance(value, dict):
        return {k: expand_secrets(v, where) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_secrets(v, where) for v in value]
    if isinstance(value, str):
        def repl(m):
            secret = os.environ.get(m.group(1))
            if secret is None:
                log_info("SECRET", f"⚠️ Tag <{m.group(1)}> sans variable dans .env{' (' + where + ')' if where else ''}")
                return m.group(0)
            return secret
        return SECRET_TAG.sub(repl, value)
    return value


def tool_notice(name):
    """Ligne d'information « outil utilisé » insérée dans le texte du chat (citation markdown, stylée dans app.py).
    Elle n'est ajoutée qu'au texte affiché : le texte envoyé à la synthèse vocale est tenu à part, donc elle n'est jamais lue."""
    return f"\n\n> 🛠️ Outil utilisé : `{name}`\n\n"
