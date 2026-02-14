import re
from datetime import datetime
from collections import deque

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
    text = text.replace('_', '').replace('`', '')
    return re.sub(r'\s+', ' ', text).strip()

def remove_emojis(text):
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002702-\U000027B0]', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)