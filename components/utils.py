import re
from datetime import datetime

# --- UTILITAIRE DE LOG ---
def log_perf(step, message=""):
    """Affiche un log de performance avec timestamp précis."""
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
    print(f"[{timestamp}] [PERF] {step.upper():<10} | {message}")

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
    """Supprime les indications de type *rit*."""
    if not text: return ""
    # Supprime uniquement les astérisques simples (pas le gras **)
    return re.sub(r'(?<!\*)\*[^*]+\*(?!\*)', '', text).strip()

def clean_text_for_tts(text):
    """Nettoyage complet pour la voix."""
    if not text: return ""
    
    # 1. SUPPRESSION TOTALE DU CODE (Indispensable pour l'audio)
    # Le flag DOTALL permet à '.' de matcher les retours à la ligne
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    
    # 2. NETTOYAGE DES ACTIONS (ex: *rit*)
    text = remove_nonverbal_cues(text)
    
    # 3. TRAITEMENT DU GRAS (On garde le texte, on enlève les **)
    # On capture le contenu entre les astérisques pour le préserver
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # 4. Autres nettoyages
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = roman_to_arabic(text)
    text = remove_emojis(text)
    
    # 5. Nettoyage final des caractères Markdown résiduels
    text = text.replace('*', '').replace('_', '').replace('`', '')
    return re.sub(r'\s+', ' ', text).strip()

def remove_emojis(text):
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002702-\U000027B0]', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)