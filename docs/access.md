# Accès à Aria pour des applications tierces

Une page web tierce peut utiliser Aria (écoute du micro, saisie texte, réponse en texte et en voix) avec une **clé API**. Chaque clé est associée à
un **profil** : ses propres serveurs MCP (outils), son message système et son LLM. Les outils et la mémoire d'Aria ne sont jamais exposés.

## Activer

1. Ajouter dans `configs/default.json` (absente ou `enabled: false` = fonction désactivée, rien n'est ouvert) :
   ```json
   "access": { "enabled": true, "port": 7861, "admin_port": 7871, "max_sessions": 8, "session_ttl_s": 900 }
   ```
2. Publier les ports dans `docker-compose.yml` : `"7861:7861"` (API, réseau local) et `"127.0.0.1:7871:7871"` (administration, ce PC seulement).
3. Créer un profil `configs/mcp/<nom>.json` (voir ci-dessous), puis redémarrer Aria.
4. Onglet **🔑 Accès tiers** : créer une clé, choisir son profil et ses **domaines autorisés** (une origine exacte par ligne, par exemple
   `https://app.example.com`). La clé n'est affichée **qu'une seule fois**.

## Profil (`configs/mcp/<nom>.json`)

```json
{ "description": "Portail RH", "system_message": "Tu aides les salariés…", "llm": "Llm_Anthropic", "voice": true,
  "mcp": { "servers": { "meteo": { "run": "uv", "args": ["run", "tools/weather_server.py"] },
                        "domotique": { "url": "http://hote:1010/mcp", "headers": { "Authorization": "Bearer <TokenDomotique>" } } } } }
```
`mcp` a le format de `config["mcp"]["params"]`. Les secrets s'écrivent `<NOM>` et viennent du `.env`, comme pour la config d'Aria. Sans profil ou avec
`"servers": {}`, la clé n'a aucun outil. `llm` désigne un module LLM **distant** de la config (Anthropic, LM Studio, OpenAI) : le LLM local est refusé.
Le serveur de mémoire personnelle n'est jamais ajouté : s'il est utile à une application, déclarer un serveur dédié (autre `MEMORY_DIR`, sans `MEMORY_UI_PORT`).

## Intégration dans une page

```html
<script src="https://ARIA:7861/aria-client.js"></script>
<script>
  const aria = AriaClient.create({ baseUrl: "https://ARIA:7861", apiKey: "ak_…", voice: true });
  aria.on("text", t => zone.textContent = t);     // texte de la réponse (cumulé)
  aria.on("state", s => console.log(s));          // idle | listening | hearing | thinking | speaking
  await aria.start();                             // écoute le micro
  aria.say("Quelle heure est-il ?");              // saisie texte
</script>
```
Méthodes : `start()`, `stop()`, `say(texte)`, `interrupt()`, `reset()`, `mute(true|false)`, `destroy()`. Événements : `state`, `text`, `user_text`,
`tool`, `audio-start`, `audio-end`, `done`, `error({code, message})`. Options : `session`, `speak`, `vad`, `modelUrl`, `cdn`.
Une page de démonstration est dans `docs/access-demo.html`.

À savoir : le **micro exige une page en HTTPS** (ou `localhost`) ; le navigateur doit avoir **accepté une fois le certificat** d'Aria (ouvrir
`https://ARIA:7861/v1/health` dans un onglet) ; le VAD est chargé depuis jsDelivr (option `cdn` pour le changer).

## API (HTTP, `Authorization: Bearer ak_…` ou `X-Api-Key`)

| Route | Rôle |
|---|---|
| `POST /v1/turn` | `{"text": "…"}` ou `{"audio": "data:audio/wav;base64,…", "session": "…", "speak": true}` ; réponse `text/event-stream` |
| `POST /v1/stop` | coupe la réponse en cours de la session |
| `POST /v1/reset` | efface l'historique de la session |
| `GET /v1/health` | vérifie la clé |
| `GET /aria-client.js`, `GET /assets/silero_vad.onnx` | fichiers publics du client |

Événements du flux : `state`, `user_text`, `text`, `tool`, `audio` (`{id, format, data, text}`, wav en base64), `done`, `error`. Une session est
identifiée par `(clé, session)` : historique et outils indépendants. Codes : 401 clé invalide, 403 origine refusée ou voix interdite, 413 trop gros,
429 débit / quota / blocage, 503 trop de sessions.

## Sécurité

Clés hachées (HMAC-SHA256), comparaison à temps constant, montrées une fois ; révocation immédiate (les sessions ouvertes sont fermées) ; expiration ;
**domaines autorisés par clé** (en-tête `Origin`, CORS limité à l'origine exacte, jamais `*`) ; débit par clé et par IP, quota de tours par jour,
blocage temporaire de l'IP après des échecs répétés ; plafonds de taille (corps 8 Mo, audio 30 s, texte 2000 caractères). L'administration est servie sur
`127.0.0.1` seulement, avec jeton et `Origin` obligatoire. **Toutes les applications consomment votre clé du LLM** : les quotas par clé en bornent le coût.
Une clé sans domaine autorisé refuse tout navigateur ; un client hors navigateur (curl, serveur) n'envoie pas d'`Origin` et n'est limité que par la clé.

## Tests

`node tests/test_aria_client.js` (client) ; dans le conteneur : `python -m unittest discover -s tests` après copie du dossier `tests/`
(les tests de l'API nécessitent `starlette`, absent du poste hôte).
