# Migration : « accès tiers » devient « agent »

Le vocabulaire d'Aria change. Ce qui s'appelait un **accès tiers** (une application ou un appareil qui utilise Aria avec une clé API) s'appelle désormais un **agent** :
une entité IA qui peut interagir avec un **interlocuteur** (un humain aujourd'hui, d'autres agents ou systèmes plus tard).

Ce guide a deux lecteurs : **les développeurs qui ont intégré Aria dans leur application** (partie 1) et **l'administrateur d'Aria** (partie 2).

---

## 1. Pour les projets qui intègrent l'API publique

### En une phrase

**Vous n'avez rien à changer.** Le client JavaScript, les routes HTTP, les en-têtes, le format des événements et votre clé API sont les mêmes ; seul le vocabulaire des
documents et de l'interface d'administration a changé.

Ce point a été vérifié par un essai automatique : une clé créée **avant** la migration, avec l'ancien format de données, est utilisée après la migration sur toutes
les routes publiques (`/v1/health`, `/v1/turn`, `/v1/reset`, `/v1/stop`, `/v1/close`, `/aria-client.js`, authentification par `Authorization` ou `X-Api-Key`) ; une
mauvaise clé reste refusée.

### Ce qui ne change pas

| Élément | Valeur (inchangée) |
|---|---|
| Client JavaScript | `<script src="https://…/aria-client.js">` et `AriaClient.create({ baseUrl, apiKey, … })` |
| Options, commandes, événements du client | `voice`, `speak`, `session`, `systemPrompt`, `speaker`, `vad`, `analyser` ; `start`, `stop`, `say`, `interrupt`, `reset`, `newSession`, `mute`, `destroy`, `analyser`, `bindState` ; `state`, `text`, `user_text`, `tool`, `audio-start`, `audio-end`, `done`, `error`, `warning`, `level` |
| Routes HTTP | `GET /v1/health`, `POST /v1/turn`, `POST /v1/stop`, `POST /v1/reset`, `POST /v1/close`, `GET /aria-client.js`, `GET /assets/…` |
| Authentification | `Authorization: Bearer ak_…` ou `X-Api-Key: ak_…` ; la clé API **n'est pas régénérée** par la migration |
| Corps de `POST /v1/turn` | `text` ou `audio`, `session`, `speak`, `system`, `speaker` |
| Flux d'événements (SSE) | `state`, `user_text`, `tool`, `text`, `audio`, `warning`, `done`, `error` |
| Codes d'erreur | `bad_json`, `bad_request`, `unauthorized`, `forbidden`, `voice_disabled`, `not_found`, `too_large`, `content_type`, `rate_limited`, `quota`, `blocked`, `session`, `busy`, `bad_audio`, `audio_too_long`, `llm`, `internal` |
| Domaines autorisés, limites de débit et de quota | ceux déjà configurés pour votre clé |
| Mémoire et conversations de votre application | conservées (même identifiant interne) |

### Ce qui a changé de nom (pour votre information)

| Avant | Maintenant | Où vous le voyez |
|---|---|---|
| « accès tiers », « application tierce », « API tierce » | **agent** | documentation, page d'administration, messages |
| « Une clé » (dans l'interface d'administration) | **un agent** et sa **clé API** | l'administrateur vous parle de « votre agent » |
| `docs/access.md` | `docs/agents.md` | documentation de l'administrateur |
| `docs/access-demo.html` | `docs/agents-demo.html` | page de démonstration |
| `docs/integration.md` | **inchangé** (titre : « Intégrer un agent Aria dans votre application web ») | ce guide d'intégration |

### Vérifications recommandées (cinq minutes)

1. Ouvrez votre application, envoyez un message texte : la réponse doit arriver comme avant.
2. Si vous utilisez la voix : `start()` puis une phrase ; vérifiez le son de la réponse.
3. Dans la console du navigateur, aucune erreur `unauthorized` ou `forbidden`.
4. Si vous aviez écrit un script qui appelle `GET /v1/health` pour surveiller le service : il continue de répondre `200`.

Si l'un de ces points échoue, **ce n'est pas la migration** : contactez l'administrateur d'Aria avec le code d'erreur (voir « Gérer les erreurs » dans `docs/integration.md`).

### Ce que vous pouvez adopter ensuite (facultatif)

- **Documentation :** reprenez le mot « agent » dans vos propres documents, pour parler comme l'administrateur.
- **`speaker`** (déjà disponible avant la migration) : si l'administrateur a autorisé votre agent à « publier les informations de son interlocuteur », vous pouvez indiquer
  la personne qui parle (`AriaClient.create({ …, speaker: { name, family } })`). Ce n'est **pas** une authentification : voir `docs/integration.md`.
- **Fermeture d'une session :** `await aria.newSession({…})` ferme proprement l'ancienne. L'administrateur peut aussi fermer une session depuis sa page.

### Si vous utilisez l'API d'administration (scripts d'exploitation)

La page d'administration (port 7871, ce PC seulement) a une API JSON protégée par un jeton, qui n'est **pas** destinée aux intégrateurs. Si vous l'avez tout de même
scriptée, ses routes `/api/keys`, `/api/keys/create`, `/api/keys/update`, `/api/keys/revoke`, `/api/keys/regenerate`, `/api/keys/delete`, `/api/profils`,
`/api/sessions` sont **inchangées** et renvoient les mêmes champs ; elles s'enrichissent seulement de `/api/sessions/close` (fermer une session) et `/api/journal*`.

---

## 2. Pour l'administrateur d'Aria

### Ce que fait la mise à jour au premier démarrage

| Élément | Avant | Après | Comment |
|---|---|---|---|
| Section de `configs/default.json` | `"access": {…}` | `"agents": {…}` | lue sous l'ancien nom tant qu'elle n'est pas renommée ; **réécrite sous le nouveau nom à la prochaine sauvegarde** de l'onglet ⚙️ Paramètres |
| Dossier de données | `models/api_access/` | `models/agents/` | **renommé une seule fois** au démarrage ; si les deux existent, le nouveau est utilisé et l'ancien n'est **pas touché** |
| Fichier des clés | `keys.json` (`{"keys": […]}`) | `agents.json` (`{"agents": […]}`) | lu dans les deux formats ; à la première écriture, l'ancien est conservé sous le nom `keys.json.migre` (jamais supprimé, jamais écrasé) |
| Secret d'installation | `secret.key` | `secret.key` (inchangé) | les clés API existantes restent valables |
| Onglet de l'interface | « 🔑 Accès tiers » | « 🤖 Agents » | |
| Section des Paramètres | « 🔑 Accès tiers : profils » | « 🤖 Agents : profils » | |
| Étiquette des logs | `[ACCESS]` | `[AGENTS]` | si vous filtrez les logs, adaptez le filtre |

Un essai sur une **copie** de votre configuration et de vos données réelles a donné : 2 agents conservés, mêmes empreintes de clés, mêmes profils.

### À faire

1. **Sauvegarder** (facultatif mais prudent) `configs/default.json` et `models/api_access/`.
2. Mettre à jour le code, puis **redémarrer** Aria (`podman restart aria`).
3. Vérifier dans l'onglet **🤖 Agents** que vos agents, leurs profils et leurs sessions sont là.
4. Ouvrir l'onglet **⚙️ Paramètres** puis **💾 SAUVEGARDER** : la config est alors réécrite avec la section `agents` (c'est ce qui rend la migration définitive).
5. Si vous avez un **docker-compose** personnalisé qui montait `models/api_access`, il n'est plus nécessaire d'en parler : le dossier `models/` est monté en entier.

### Retour en arrière

La migration ne détruit rien : pour revenir à l'ancienne version, rendez à `models/agents/` le nom `models/api_access/`, renommez `keys.json.migre` en `keys.json`
et supprimez `agents.json`. Les clés API d'origine fonctionnent alors comme avant.

### Ce qui n'a pas été renommé, volontairement

| Élément | Pourquoi |
|---|---|
| Routes `/v1/…`, `aria-client.js`, `AriaClient` | c'est le contrat public : le changer casserait tous les projets pour rien |
| En-têtes `X-Memory-*`, préfixe `ak_` des clés | internes à Aria, ou secrets déjà distribués |
| Type de personne `systeme` dans la mémoire | c'est un format de données ; le mot « agent » est celui de l'interface |
| Port `7861` (API), `7871` (administration) | inchangés |
