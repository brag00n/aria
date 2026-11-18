# Intégrer un agent Aria dans votre application web

Ce guide s'adresse aux développeurs d'une application web qui veulent offrir à leurs utilisateurs un assistant vocal et textuel : écoute du micro,
saisie de texte, réponse en texte et en voix. Votre application est un **agent** d'Aria : une entité IA qui dialogue avec un interlocuteur (aujourd'hui un humain).

> **Vous aviez déjà intégré Aria ?** Rien ne casse : le client JavaScript, les routes `/v1/…` et votre clé API sont inchangés. Voir `docs/migration-agents.md`. Vous ajoutez un script à votre page ; l'interface (boutons, visuels, styles) reste **entièrement la vôtre**.

Ce que vous devez obtenir de l'administrateur d'Aria :

| Élément | Exemple | Rôle |
|---|---|---|
| Adresse du service | `https://aria.exemple.fr` | `baseUrl` du client |
| Clé API | `ak_…` | identifie votre agent ; elle est montrée une seule fois, **gardez-la secrète côté serveur si possible** (voir « Sécurité ») |
| Domaines autorisés | `https://app.exemple.fr` | votre page doit être servie depuis l'un d'eux |

## Démarrage en cinq minutes

```html
<button id="micro">🎙</button>
<input id="champ" placeholder="Écrire à l'assistant">
<div id="reponse"></div>

<script src="https://aria.exemple.fr/aria-client.js"></script>
<script>
  const aria = AriaClient.create({
    baseUrl: "https://aria.exemple.fr",
    apiKey: "ak_…"
  });

  aria.on("text",  t => reponse.textContent = t);                // texte de la réponse (cumulé)
  aria.on("error", e => reponse.textContent = "Erreur : " + e.message);

  micro.onclick = async () => {
    try { aria.listening ? aria.stop() : await aria.start(); }  // micro : une page HTTPS est nécessaire
    catch (e) { reponse.textContent = "Micro impossible : " + e.message; }
  };
  champ.onkeydown = e => { if (e.key === "Enter") { aria.say(champ.value); champ.value = ""; } };
</script>
```

Un exemple complet (halo animé, forme d'onde, boutons) est fourni dans `agents-demo.html`.

## Avant de commencer : trois conditions du navigateur

1. **HTTPS pour le micro.** Un navigateur n'ouvre le micro que sur une page en `https://` (ou `http://localhost` pendant le développement). La saisie texte n'a pas cette contrainte.
2. **Certificat accepté.** Si le service d'Aria utilise un certificat interne (auto-signé), chaque navigateur doit l'accepter une fois : ouvrez
   `https://aria.exemple.fr/v1/health` dans un onglet et confirmez. Sans cela, les appels échouent avec une erreur réseau.
3. **Geste de l'utilisateur pour le son.** Les navigateurs interdisent de jouer un son avant un clic ou une touche. Déclenchez `start()` ou `say()` depuis un gestionnaire de clic.

## Les options de `AriaClient.create({…})`

| Option | Défaut | Rôle |
|---|---|---|
| `baseUrl`, `apiKey` | obligatoires | adresse du service et clé API |
| `voice` | `true` | `false` : texte seulement, `start()` est refusé |
| `speak` | `true` | demander la réponse en voix ; `mute(true)` la coupe ensuite sans changer l'option |
| `session` | `"default"` | identifiant de conversation. Deux identifiants différents ont chacun leur historique. Utilisez-en un par utilisateur final de votre page |
| `systemPrompt` | aucun | consignes de votre application pour l'assistant (voir plus bas) |
| `speaker` | aucun | `{ name: "Léa", family: "Dupont" }` : la personne qui parle, si l'administrateur d'Aria a autorisé votre agent à « publier les informations de son interlocuteur ». Fixée à la création de la session. **Ce n'est pas une authentification** : Aria vous croit sur parole. Ignoré par une clé sans ce droit |
| `vad` | `{}` | sensibilité de l'écoute : `positive` (0,8), `negative` (0,4), `redemption` (20 trames de silence), `minFrames` (3) |
| `analyser` | `false` | prépare d'emblée l'analyse du son pour vos visuels (sinon, à la première utilisation) |
| `debug` | `false` | écrit dans la console l'état de l'écoute : utile quand le micro ne réagit pas |
| `cdn`, `modelUrl` | adresses par défaut | emplacement des bibliothèques de détection de parole, si vous voulez les héberger vous-même |

## Les commandes

| Méthode | Effet |
|---|---|
| `await start()` | ouvre le micro et écoute ; chaque phrase détectée est envoyée à l'assistant |
| `stop()` | arrête l'écoute ; une réponse en cours continue |
| `say(texte)` | envoie une saisie texte |
| `interrupt()` | coupe la réponse en cours (voix et génération) |
| `reset()` | efface l'historique de la conversation ; les consignes `systemPrompt` sont conservées |
| `await newSession({systemPrompt, speaker, session})` | nouvelle conversation, avec de **nouvelles consignes** si `systemPrompt` est fourni et une **autre personne** si `speaker` l'est (`speaker: null` l'efface). Renvoie l'identifiant de session |
| `mute(true)` / `mute(false)` | coupe ou rétablit la voix (le texte continue) |
| `destroy()` | libère le micro, la lecture et l'analyse ; le client est inutilisable ensuite |
| `on(nom, fn)`, `off(nom, fn)` | s'abonne ou se désabonne d'un événement |
| `analyser("voice")`, `analyser("mic")` | `AnalyserNode` Web Audio pour vos visuels (voir « Habiller votre interface ») |
| `bindState(élément)` | habillage en CSS pur ; renvoie la fonction qui le retire |
| `state`, `listening`, `muted`, `session` | état courant (lecture seule) |

## Les événements

| Événement | Données | Quand |
|---|---|---|
| `state` | `"idle"`, `"listening"`, `"hearing"`, `"thinking"`, `"speaking"` | à chaque changement d'état |
| `text` | texte cumulé de la réponse | pendant la génération : **remplacez** l'affichage, n'ajoutez pas |
| `user_text` | ce que l'utilisateur a dit, transcrit | après une phrase dite |
| `tool` | nom d'un outil utilisé par l'assistant | une fois par outil et par tour |
| `audio-start` | `{id, text, element}` | début d'un bloc de voix : `text` est la phrase dite, `element` l'élément `<audio>` |
| `audio-end` | rien | fin de la voix de la réponse |
| `level` | `{source: "mic" ou "voice", value: 0 à 1}` | niveau sonore, toutes les 50 ms |
| `done` | `{ms}`, `{stopped:true}`, `{empty:true}`, `{ignored:true}` ou `{stopped:true, reason:"keyword"}` | fin d'un tour |
| `conversation` | `{state: "open" ou "closed"}` | **seulement si l'administrateur a activé les mots d'ouverture/clôture pour votre agent** : une conversation vient de s'ouvrir ou de se fermer |
| `warning` | `{code, message}` | avertissement sans gravité (par exemple `system_ignored`) |
| `error` | `{code, message}` | erreur (voir « Gérer les erreurs ») |

Les états : `idle` (rien en cours), `listening` (micro ouvert, silence), `hearing` (l'utilisateur parle), `thinking` (l'assistant prépare sa réponse),
`speaking` (l'assistant parle). Si l'utilisateur parle pendant que l'assistant parle, la voix de l'assistant baisse, puis s'arrête quand la phrase est comprise.

## Habiller votre interface

Le client ne dessine rien. Quatre façons de l'habiller, de la plus simple à la plus libre :

**1. En CSS seulement : `bindState(élément)`.** Le client écrit sur votre élément l'attribut `data-aria-state` et la variable CSS `--aria-level`
(0 à 1, niveau sonore de la voix en cours). Tout le style est le vôtre :

```html
<div id="halo"></div>
<style>
  #halo { --aria-level: 0; width: 96px; height: 96px; border-radius: 50%; border: 3px solid #ccc;
          box-shadow: 0 0 calc(8px + var(--aria-level) * 40px) #2a6df4; }
  #halo[data-aria-state="listening"] { border-color: #16a34a; }
  #halo[data-aria-state="thinking"]  { animation: pulse 1s infinite; }
  #halo[data-aria-state="speaking"]  { border-color: #2a6df4; }
  @keyframes pulse { 50% { transform: scale(1.06); } }
</style>
<script> aria.bindState(document.getElementById("halo")); </script>
```
`bindState(el, {level: false})` ne pose que l'état, sans mesure du niveau sonore.

**2. Un niveau sonore : l'événement `level`.** Pour une jauge, un avatar ou une bouche animée :
```js
aria.on("level", ({source, value}) => { if (source === "voice") bouche.style.transform = `scaleY(${0.2 + value})`; });
```
`"voice"` est la voix de l'assistant (un dernier `0` signale la fin), `"mic"` celle de l'utilisateur pendant l'écoute.

**3. Une forme d'onde ou un spectre : `analyser()`.** Renvoie un `AnalyserNode` Web Audio standard, à lire dans votre boucle d'animation :
```js
const an = aria.analyser("voice");               // "mic" : null tant que l'écoute n'est pas démarrée
const data = new Uint8Array(an.fftSize);
(function dessiner() { an.getByteTimeDomainData(data); /* dessiner sur un canvas */ requestAnimationFrame(dessiner); })();
```
Le son de l'assistant reste audible. Le micro n'est jamais renvoyé vers les haut-parleurs. L'analyse n'est installée que si vous la demandez.

**4. Suivre la phrase dite : `audio-start`.** `text` donne la phrase en cours de lecture (sous-titres, surbrillance), `element` l'élément `<audio>` du bloc.

## Donner des consignes à l'assistant (`systemPrompt`)

Votre application peut décrire à l'assistant son contexte : nom du site, ton, sujets, page en cours.

```js
const aria = AriaClient.create({
  baseUrl: "https://aria.exemple.fr", apiKey: "ak_…",
  session: "utilisateur-42",
  systemPrompt: "Tu es l'assistante du site Bleu, une boutique de vélos. Tu tutoies. L'utilisateur est sur la page « Entretien »."
});
```
- **Vos consignes complètent celles qui sont attachées à votre agent** ; elles ne les remplacent pas. L'administrateur peut y avoir fixé des limites (sujets, outils
  disponibles) que vos consignes ne peuvent pas lever.
- **Elles sont fixées à la création de la conversation.** Si vous les modifiez en cours de route, l'événement `warning` (`code: "system_ignored"`) vous prévient
  qu'elles sont ignorées. Pour repartir avec de nouvelles consignes : `await aria.newSession({ systemPrompt: "nouvelles consignes" })`.
  `reset()` efface l'historique mais **garde** les consignes.
- Limite : **4000 caractères**, du texte seulement.
- **N'y mettez aucun secret** : le texte passe par le navigateur de l'utilisateur.

## Une conversation par utilisateur

Une conversation est identifiée par la paire (votre agent, `session`). Donnez à chaque utilisateur de votre page un identifiant stable et sans donnée personnelle
(par exemple un identifiant interne opaque) : il retrouve son historique, et deux utilisateurs ne se mélangent jamais. Une conversation inactive est fermée
au bout d'un moment ; la suivante repart d'un historique vide. Le nombre de conversations simultanées est limité.

## Gérer les erreurs

L'événement `error` donne `{code, message}`. Les messages sont lisibles par l'utilisateur ; testez le `code` dans votre logique.

| Code | Signification | Que faire |
|---|---|---|
| `network` | service injoignable, ou certificat non accepté | vérifier l'adresse et le certificat (voir « Avant de commencer ») |
| `unauthorized` | clé invalide, révoquée ou expirée | contacter l'administrateur ; ne pas réessayer en boucle |
| `forbidden` | l'adresse de votre page n'est pas autorisée pour cet agent | demander l'ajout de votre domaine |
| `voice_disabled` | la voix n'est pas permise pour cet agent | utiliser `voice: false` |
| `rate_limited` | trop de requêtes | attendre (l'en-tête `Retry-After` donne le délai en secondes) |
| `quota` | limite quotidienne de tours atteinte | informer l'utilisateur ; reprise le lendemain |
| `blocked` | trop d'échecs d'authentification depuis cette adresse | attendre quelques minutes |
| `too_large` | texte ou consignes trop longs, ou requête trop volumineuse | raccourcir |
| `audio_too_long` | phrase de plus de 30 secondes | la reformuler en plusieurs phrases |
| `bad_audio`, `bad_request`, `bad_json` | requête mal formée | signaler un défaut d'intégration |
| `busy` | un tour est déjà en cours dans cette conversation | attendre la fin ou appeler `interrupt()` |
| `session` | conversation impossible (trop de conversations ouvertes, ou agent pas encore configuré) | réessayer plus tard, sinon contacter l'administrateur |
| `llm` | l'assistant n'a pas pu répondre | proposer de réessayer |
| `internal` | erreur interne | réessayer ; contacter l'administrateur si cela persiste |

Après une erreur, le client revient à l'état `idle` (ou `listening` si l'écoute est active) et reste utilisable.

## Sécurité : à retenir

- **La clé est visible dans votre page**, donc de quiconque l'ouvre. Elle est protégée par la liste des **domaines autorisés** (une page d'un autre domaine ne peut pas
  s'en servir dans un navigateur) et par des limites de débit et de quota. Ne la réutilisez pas ailleurs ; demandez-en une par application et par environnement.
- Si la clé a fuité : demandez sa **révocation** (effective immédiatement) et une nouvelle clé.
- Pour une application où la clé ne doit jamais être visible, appelez le service depuis **votre serveur** avec l'API HTTP ci-dessous (la clé reste chez vous).
- Affichez les réponses avec `textContent`, **jamais** `innerHTML` : le texte vient d'un modèle de langage.
- Ne confiez pas à l'assistant de données dont vos utilisateurs ne devraient pas disposer : il répond avec ce que contient la conversation et ce que ses outils lui donnent.

## Utiliser l'API HTTP directement (sans le client JavaScript)

Authentification : en-tête `Authorization: Bearer ak_…` (ou `X-Api-Key: ak_…`). Corps en JSON (`Content-Type: application/json`).

| Route | Rôle |
|---|---|
| `POST /v1/turn` | un tour de conversation ; la réponse est un flux `text/event-stream` |
| `POST /v1/stop` | coupe la réponse en cours de la conversation |
| `POST /v1/reset` | efface l'historique de la conversation |
| `POST /v1/close` | ferme la conversation et libère sa place |
| `GET /v1/health` | vérifie la clé |
| `GET /aria-client.js` | le script client |

Corps de `POST /v1/turn` :
```json
{ "text": "Quelle heure est-il ?", "session": "utilisateur-42", "speak": true, "system": "consignes de l'application", "speaker": {"name": "Léa", "family": "Dupont"} }
```
ou `"audio": "data:audio/wav;base64,…"` à la place de `text` (WAV mono, 30 secondes au plus). Tous les champs sauf `text` ou `audio` sont facultatifs.
`speaker` n'a d'effet que pour un agent autorisé à publier les informations de son interlocuteur, et seulement à la création de la session.
Limites : texte de 2000 caractères, requête de 8 Mo, consignes de 4000 caractères.

Événements du flux (`event:` et `data:` en JSON) :
```
event: state    data: {"state": "thinking"}
event: user_text data: {"text": "…"}                         (tour vocal : ce qui a été compris)
event: tool     data: {"name": "meteo"}
event: text     data: {"text": "Il est quatorze heures…"}    (texte cumulé)
event: audio    data: {"id": "…", "format": "wav", "data": "<base64>", "text": "Il est quatorze heures."}
event: warning  data: {"code": "system_ignored", "message": "…"}
event: done     data: {"ms": 1830}
event: error    data: {"code": "llm", "message": "…"}
```
Exemple depuis un serveur :
```bash
curl -N https://aria.exemple.fr/v1/turn \
  -H "Authorization: Bearer $ARIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Bonjour", "session": "demo", "speak": false}'
```
Les erreurs de l'API (avant le flux) sont un JSON `{"error": {"code": "…", "message": "…"}}` avec le statut HTTP correspondant : 400, 401, 403, 413, 415, 429, 503.
Un client qui n'est pas un navigateur n'envoie pas d'en-tête `Origin` : seules la clé, le débit et le quota le limitent.
