# Agents : applications et appareils qui dialoguent avec Aria

> **Ce document s'adresse à l'administrateur d'Aria** (configuration, agents, profils, sécurité). Le guide destiné aux développeurs des
> applications qui intègrent un agent, sans détail d'installation, est `docs/integration.md` : c'est celui à leur remettre avec la clé API de l'agent.
> **Vous veniez de l'ancien vocabulaire « accès tiers » ?** Voir `docs/migration-agents.md`.

Un **agent** est une entité IA qui peut interagir avec un **interlocuteur** (un humain aujourd'hui, d'autres agents ou systèmes demain). Concrètement : une page web,
une application ou un appareil (un robot) qui utilise Aria (écoute du micro, saisie texte, réponse en texte et en voix) avec une **clé API**. Chaque agent a
un **profil** : ses propres serveurs MCP (outils), son message système et son LLM. Les outils d'Aria ne sont jamais exposés ; sa mémoire ne l'est que par un profil qui la référence (voir « Mémoire des agents »).

## Activer

1. Ajouter dans `configs/default.json` (absente ou `enabled: false` = fonction désactivée, rien n'est ouvert) :
   ```json
   "agents": { "enabled": true, "port": 7861, "admin_port": 7871, "max_sessions": 8, "session_ttl_s": 900 }
   ```
2. Publier les ports dans `docker-compose.yml` : `"7861:7861"` (API, réseau local) et `"127.0.0.1:7871:7871"` (administration, ce PC seulement).
3. Redémarrer Aria, puis créer un profil dans l'onglet **⚙️ Paramètres**, section « 🤖 Agents : profils » (voir ci-dessous).
4. Onglet **🤖 Agents** : créer un agent, choisir son profil et ses **domaines autorisés** (une origine exacte par ligne, par exemple
   `https://app.example.com`). Sa clé API n'est affichée **qu'une seule fois**.

> **Données.** Les agents sont dans `models/agents/agents.json` (clés API hachées, jamais en clair) avec le secret d'installation `secret.key`. Au premier démarrage
> après la mise à jour, l'ancien dossier `models/api_access/` est **renommé** et l'ancien `keys.json` lu puis conservé sous le nom `keys.json.migre` : rien n'est perdu et
> les clés API déjà distribuées continuent de fonctionner.

## Profils (onglet ⚙️ Paramètres)

Un profil décide de ce qu'un agent peut utiliser. Les profils sont dans `configs/default.json`, section `agents.profiles`, et se modifient dans l'onglet
**⚙️ Paramètres**, section « 🤖 Agents : profils » (puis « 💾 SAUVEGARDER »). Un profil **référence** des serveurs MCP déjà déclarés dans la section
« Outils MCP » : il n'en redéclare aucun et ne copie aucun secret.

```json
"agents": {
  "enabled": true,
  "private_servers": [],
  "profiles": {
    "portail": {
      "description": "Portail RH",
      "system_message": "Tu aides les salariés…",
      "llm": "Llm_Anthropic",
      "voice": true,
      "servers": ["meteo_local", "recherche_web"]
    }
  }
}
```

| Clé du profil | Rôle |
|---|---|
| `servers` | liste de **noms de serveurs MCP** déclarés dans `mcp.params.servers` : tous leurs outils sont disponibles pour la clé. Vide ou absent : aucun outil |
| `system_message` | règles de base de l'assistant pour cette clé (les consignes `systemPrompt` de l'application s'y ajoutent, sans les remplacer) |
| `llm` | **nom d'un module** de la section `llm` de la config (`Llm_Anthropic`, `Llm_Ministral_ollama`…) ; vide = le module actif d'Aria. Le LLM local est refusé sans le droit « LLM local ». Voir « Un LLM par agent » |
| `voice` | `false` : les clés de ce profil n'ont pas la voix |
| `description` | texte affiché dans l'administration |
| `conversation` | mots d'ouverture, de clôture et d'arrêt (voir « Conversation à mots-clés »). Absent = désactivé : l'agent répond à tout |

- **Clé API de l'agent.** « 🔄 Régénérer » (page d'administration) donne une nouvelle clé API au **même agent** : l'identifiant interne, les réglages, les domaines et la
  mémoire sont conservés, l'ancienne clé cesse de fonctionner immédiatement et ses conversations ouvertes sont fermées. À utiliser en cas de fuite de la clé.
- **Serveurs privés.** Les serveurs de `agents.private_servers` (aucun par défaut) ne peuvent jamais figurer dans un profil : la sauvegarde est refusée.
  À utiliser pour fermer un serveur sensible (par exemple la domotique) à tous les agents.
- **Mémoire.** Le serveur `memoire` peut figurer dans un profil, à condition de déclarer son canal HTTP (`MEMORY_HTTP_PORT` dans son `env`, voir plus bas).
  Un serveur de mémoire sans canal HTTP est refusé : un agent ne lance jamais son propre processus de mémoire.
- **Validation à la sauvegarde.** Noms de profil (lettres, chiffres, `-`, `_`), types, **clés inconnues** (une faute de frappe est signalée), serveurs inconnus
  ou privés, LLM local. Les profils sont contrôlés contre les serveurs et LLM de la configuration **enregistrée avec la même sauvegarde** : retirer un serveur
  encore utilisé par un profil est refusé. En cas d'erreur, rien n'est modifié, ni en mémoire ni dans le fichier.
- **Appliqué sans redémarrage.** Les profils sont relus à chaque création de conversation : un profil modifié vaut pour les **nouvelles** conversations. Les
  conversations déjà ouvertes gardent l'ancien profil jusqu'à leur expiration ; la case « Fermer les conversations ouvertes des agents dont le profil change »
  les ferme tout de suite (elles repartent avec le profil à jour au prochain échange). Les autres sections de l'onglet exigent toujours « 🔄 Redémarrer ».
- **Profil supprimé ou renommé.** Un agent qui l'utilise reçoit une erreur à la création de sa prochaine conversation. La sauvegarde affiche un avertissement
  qui liste ces agents : changer leur profil dans l'onglet « 🤖 Agents ».
- **Secrets.** Ils restent dans la déclaration du serveur (`env`, `headers`, tags `<NOM>` résolus depuis le `.env`). L'administration n'affiche des profils que
  le nom et le type (HTTP ou local) des serveurs.
- **Granularité.** On choisit des **serveurs**, pas des outils : un serveur qui expose seize outils les donne tous. Pour n'exposer qu'une partie, déclarer un
  serveur MCP dédié.

## Droits des agents et agent par défaut « Aria »

Un **droit** est une capacité qu'Aria a toujours eue et qu'un agent n'a pas par défaut. Ils s'accordent un par un dans la fiche d'un agent (page d'administration, onglet **🤖 Agents**).

| Droit | Ce qu'il permet |
|---|---|
| **LLM local** | utiliser un moteur de langage local (llama-cpp), en plus des LLM distants. Le profil seul ne suffit pas : c'est l'agent qui doit avoir le droit |
| **Tous les serveurs MCP** | disposer de **tous** les serveurs MCP de la configuration, **serveurs privés compris**, au lieu de ceux de son profil |
| **Interlocuteur global et mémoire d'Aria** | parler « comme Aria » : interlocuteur global (la liste déroulante), mémoire du foyer, souvenirs partagés, fiches créées sans validation |
| **Sans limite de débit ni de quota** | aucune limite de requêtes par minute ni de tours par jour |
| **Publier ses propres liens** | (existant) ses liens avec une personne deviennent publics |
| **Publier les informations de son interlocuteur** | (existant) agit au nom de la personne qui lui parle |

Aucun droit n'est accordé par défaut : un agent existant **ne gagne aucun privilège** à la mise à jour.

**L'agent par défaut `aria`.** Au premier démarrage, Aria crée l'agent `aria` (identifiant fixe). Il représente l'assistant lui-même :

- **tous ses droits sont activés et verrouillés** (cases cochées et grisées dans la page) ; ils ne se modifient pas, et ne peuvent pas être retirés par erreur ;
- il **ne se supprime pas et ne se révoque pas** (le chat de l'interface en dépend) ; sa clé API peut être **régénérée** (pour un client externe) ;
- il **réutilise le moteur et le serveur MCP d'Aria** (interlocuteur global, mémoire du foyer) : il ne lance aucun second processus ;
- son profil est **implicite** : il suit la configuration d'Aria (LLM actif, conversation à mots-clés) et référence tous les serveurs. Pour le personnaliser, écrire un profil nommé `aria` dans `agents.profiles` : il l'emporte ;
- il existe **même si `agents.enabled` est faux** : seul le port 7861 est alors fermé.

Un autre agent peut avoir les **mêmes droits** qu'Aria : en cochant les quatre droits ci-dessus.

> **Pas d'authentification de la personne.** Les droits « interlocuteur global » et « publier les informations de son interlocuteur » reposent sur ce que l'application ou l'utilisateur déclare. À réserver aux applications de confiance.

### Le chat de l'interface passe par l'agent `aria`

Le chat du port 7860 peut utiliser l'agent `aria` **en interne** (même processus, aucun HTTP, aucune clé exposée dans la page), avec le même cœur de tour que l'API.
Il est piloté par un drapeau de la configuration, **désactivé par défaut** :

```json
"agents": { "chat_via_agent": true }
```

- **Désactivé** (défaut) : le chat utilise l'ancien chemin, inchangé.
- **Activé** : texte, voix, arrêt, nouvelle conversation, mots-clés, liste d'interlocuteurs et journal d'audit passent par l'agent. Le drapeau est relu **à chaque évènement** : on peut revenir en arrière sans redémarrer.
- Le journal d'audit garde `src = aria:Aria` et `who` = l'interlocuteur choisi dans la liste ; la conversation de l'interface est la session `aria:ui`.
- Pour l'essayer : passer le drapeau à `true`, puis suivre la **liste de vérification** ci-dessous.

**Liste de vérification (aucun test automatique de l'interface Gradio n'existe)** : texte saisi ; voix (micro, VAD) ; arrêt par le bouton et par un mot ; ouverture et clôture par mots-clés ; changement d'interlocuteur dans la liste ; souvenir écrit au nom de l'interlocuteur choisi ; voix qui se lit en continu ; « Nouvelle conversation » ; ligne d'audit (`src`, `who`, `conv`) ; `podman restart aria` puis nouvelle conversation.

## Un LLM par agent

Chaque profil nomme son module LLM, et chaque module porte **sa propre** clé, son modèle et son URL. Deux modules du même type avec des réglages différents donnent donc deux agents qui ne partagent rien :

```json
"llm": {
  "Llm_Anthropic":      {"params": {"backend_type": "anthropic", "apikey": "<CLE_ANTHROPIC_A>", "model_name": "claude-haiku-5-5"}},
  "Llm_Anthropic_Pro":  {"params": {"backend_type": "anthropic", "apikey": "<CLE_ANTHROPIC_B>", "model_name": "claude-opus-5-5", "base_url": "<URL_B>"}}
},
"agents": {"profiles": {
  "rapide": {"llm": "Llm_Anthropic"},
  "expert": {"llm": "Llm_Anthropic_Pro"}
}}
```

- `apikey`, `base_url` : écrire une balise `<NOM>` dont la valeur est dans `.env` (jamais la clé en clair). Une balise sans variable reste littérale et le moteur le signale dans les logs ; `apikey` vide = variable `LLM_ANTHROPIC_DEFAULT_APIKEY` (Anthropic seulement).
- `model_name` : celui du module, sans repli sur un autre module.
- Un module inconnu est refusé à la sauvegarde du profil (« LLM … inconnu »).

## Mémoire des agents

Il n'y a qu'**un seul serveur de mémoire** (`memoire`) : c'est lui qui garantit l'intégrité et la confidentialité des données. Chaque agent y est une
**personne de type système**, liée à la **clé interne de l'accès** (`k_xxxx`) : renommer l'accès ou **régénérer sa clé API** ne fait pas perdre sa mémoire.

| | Un agent (système) | Une personne (Aria, page de gestion) |
|---|---|---|
| Personnes et relations du foyer | **toutes**, en lecture | toutes |
| Souvenirs | **les siens** + les souvenirs partagés de type `foyer` | les siens + le partagé |
| Ce qu'il écrit | toujours **privé** (même de type `foyer`) | comme avant |
| Liens qu'il crée | seulement entre **lui** et une personne ; **privés** par défaut, **publics** avec l'option de sa clé « publier ses propres liens » (voir ci-dessous) | tous |
| Changer d'identité | **impossible** (imposé par le serveur) | oui |

**Publier ses propres liens (option par clé, désactivée par défaut).** Case « Peut publier ses propres liens » dans l'édition d'une clé (champ `publish_links`).
Pour un accès qui l'a, un lien dont il est l'une des deux extrémités (« moi est le propriétaire d'Olivier ») est **public** : il apparaît dans le graphe, la page
Mémoire et les réponses d'Aria. Garde-fous, imposés par le serveur de mémoire (le LLM ne peut ni activer l'option ni la contourner) :

- **jamais de lien entre d'autres personnes** (l'une des deux extrémités doit être l'accès lui-même) ;
- **une personne nouvelle peut être créée**, mais sa fiche est marquée **« à valider »** (⚠ dans la page Mémoire, case « Validée » dans sa fiche) et reste « citée » (hors de la liste des interlocuteurs d'Aria) ; un accès ne crée que **5 fiches par 24 h** ;
- **mise à jour limitée à ses propres liens** : le lien qu'un humain ou un autre accès a établi entre les mêmes deux personnes n'est jamais modifié ;
- il peut **oublier ses liens publics**, jamais ceux des autres ; un agent ne renseigne pas le sexe d'une fiche ;
- chaque lien et chaque fiche gardent leur **trace d'audit** (l'échange d'origine) ; un humain peut tout supprimer dans la page Mémoire ;
- si plusieurs personnes portent le même prénom, il est invité à préciser le nom de famille (rien n'est deviné).

**Publier les informations de son interlocuteur (option par clé, désactivée par défaut, indépendante de la précédente).** Case « Peut publier les informations de
son interlocuteur » (champ `publish_speaker`). L'accès agit alors **au nom de la personne qui lui parle**, avec les droits d'Aria : ce que la personne dit va dans
**ses** souvenirs (privés), l'accès lit les siens et ceux du foyer, relie n'importe quelles personnes (liens publics), crée des fiches **sans validation** et peut oublier
les souvenirs de cette personne. Ses propres constats (mesures, état d'un appareil) restent dans **sa** mémoire privée, par l'outil séparé `noter_observation`.

- **L'interlocuteur est propre à la session.** Il n'est jamais celui d'Aria : un agent ne change pas à qui parle l'interface.
- **Il est désigné de deux façons** : par l'application (`speaker: {name, family}` à la création de la session, voir `docs/integration.md`) ; sinon par le LLM, qui reçoit l'outil `identifier_interlocuteur` et demande le prénom comme le fait Aria. Quand l'application l'a fourni, le LLM **ne peut plus le changer**.
- **La personne n'est pas authentifiée** (prévu plus tard) : l'application, ou ce que dit l'utilisateur, est cru sur parole. Celui qui peut parler à l'application peut donc se
  faire passer pour n'importe quelle personne du foyer, lire ses souvenirs et en écrire à son nom. **À réserver aux applications de confiance de votre réseau.** Tout reste
  tracé : le journal d'audit et la trace de chaque souvenir portent l'accès **et** la personne (`who`).
- Un accès qui n'a pas ce droit ignore le champ `speaker` (il n'est même pas examiné).

Un agent du foyer connaît donc les noms de toutes les personnes et leurs liens, mais jamais leurs souvenirs privés. Pour un agent **hors foyer**, ne pas lui
donner la mémoire d'Aria : lui fournir un serveur de mémoire dédié (même programme, autre `MEMORY_DIR`, son propre `MEMORY_HTTP_PORT`).

**Tiers hors foyer.** Déclarer un second serveur de mémoire (même programme, dossier distinct, port HTTP distinct, sans `MEMORY_UI_PORT`) et le lister dans
`agents.hosted_only` : Aria le démarre pour les agents, mais **n'en expose jamais les outils à l'interface** (ils porteraient les mêmes noms que ceux de `memoire`
et les écraseraient). Les profils de ces agents le référencent comme n'importe quel serveur. Toutes les conversations d'un même profil partagent **un seul processus**
et la mémoire de leur accès. Deux serveurs de mémoire sur le même dossier ou le même port sont refusés à la sauvegarde (ils se verrouilleraient).

```json
"mcp": { "params": { "servers": {
  "memoire":      { "run": "uv", "args": ["run", "/aria/tools/memory/server.py"], "env": { "MEMORY_DIR": "/aria/models/memory",      "MEMORY_UI_PORT": "7870", "MEMORY_HTTP_PORT": "7872" } },
  "memoire_acme": { "run": "uv", "args": ["run", "/aria/tools/memory/server.py"], "env": { "MEMORY_DIR": "/aria/models/memoire_acme", "MEMORY_HTTP_PORT": "7873" } } } } },
"agents": { "hosted_only": ["memoire_acme"], "profiles": { "acme": { "servers": ["memoire_acme"] } } }
```

Mise en place pour le foyer : ajouter `MEMORY_HTTP_PORT` (par exemple `7872`) à l'`env` du serveur `memoire`, puis référencer `memoire` dans le profil. Le canal est ouvert
par le **même processus** que celui d'Aria, sur `127.0.0.1` seulement, et protégé par un secret **généré à chaque démarrage d'Aria** (jamais écrit dans un
fichier ni dans `default.json`) ; sans lui, le canal ne s'ouvre pas. Le conteneur n'a pas à publier ce port.

## Intégration dans une page

Le client JavaScript n'impose **aucune interface** : pas de bouton, pas de style. La page de l'agent dessine ses propres commandes et sa propre
visualisation avec son CSS, et pilote Aria par les méthodes et les événements ci-dessous.

### Démarrage minimal

```html
<script src="https://ARIA:7861/aria-client.js"></script>
<script>
  const aria = AriaClient.create({ baseUrl: "https://ARIA:7861", apiKey: "ak_…" });
  aria.on("text", t => reponse.textContent = t);          // texte de la réponse (cumulé)
  monBoutonMicro.onclick = () => aria.listening ? aria.stop() : aria.start();
  monChamp.onkeydown = e => { if (e.key === "Enter") aria.say(monChamp.value); };
</script>
```
`start()` est asynchrone et peut échouer (micro refusé, VAD introuvable) : l'entourer d'un `try/catch`. Les écritures dans la page se font en `textContent`,
jamais en `innerHTML` : le texte vient d'un modèle de langage.

### Options de `AriaClient.create({…})`

| Option | Défaut | Rôle |
|---|---|---|
| `baseUrl`, `apiKey` | obligatoires | adresse de l'API d'Aria et clé API |
| `voice` | `true` | `false` : texte seulement, `start()` est refusé |
| `speak` | `true` | demander la réponse en voix ; `mute(true)` la coupe sans changer l'option |
| `session` | `"default"` | identifiant de session : historique et outils indépendants (un par utilisateur final de la page, par exemple) |
| `systemPrompt` | aucun | consignes de l'application pour le LLM, fixées à la création de la session (voir « Prompt système de l'application ») |
| `vad` | `{}` | sensibilité de l'écoute : `positive` (0,8), `negative` (0,4), `redemption` (20 trames de silence), `minFrames` (3) |
| `modelUrl` | servi par Aria | adresse du modèle Silero du VAD |
| `cdn` | jsDelivr | `{ort, vad}` : adresses des deux bibliothèques du VAD, pour les héberger ailleurs |
| `analyser` | `false` | prépare d'emblée l'analyse du son (sinon elle l'est dès qu'on s'abonne à `level` ou qu'on appelle `analyser()`) |
| `debug` | `false` | trace dans la console l'état, le VAD (probabilité de parole par seconde) et les envois |

### Commandes

| Méthode | Effet |
|---|---|
| `await start()` | ouvre le micro et écoute (la page doit être en HTTPS ou sur `localhost`) |
| `stop()` | arrête l'écoute ; une réponse en cours continue |
| `say(texte)` | envoie une saisie texte |
| `interrupt()` | coupe la réponse en cours (lecture et génération) |
| `reset()` | efface l'historique de la session ; ses consignes (`systemPrompt`) sont conservées |
| `await newSession({systemPrompt, session})` | nouvelle conversation : ferme la session actuelle et en ouvre une autre au prochain tour, avec de **nouvelles consignes** si `systemPrompt` est fourni (sinon les mêmes) ; `session` impose un identifiant, sinon il est généré. Renvoie l'identifiant |
| `mute(true)` / `mute(false)` | coupe ou rétablit la voix d'Aria (le texte continue) |
| `destroy()` | libère le micro, la lecture et l'analyse ; le client est inutilisable ensuite |
| `on(nom, fn)`, `off(nom, fn)` | abonnement aux événements |
| `analyser("voice")` ou `analyser("mic")` | `AnalyserNode` Web Audio (voir plus bas) |
| `bindState(élément)` | habillage en CSS pur (voir plus bas) ; renvoie la fonction qui le retire |
| `state`, `listening`, `muted`, `session` | état courant et identifiant de session (lecture seule) |

### Événements

| Événement | Données | Quand |
|---|---|---|
| `state` | `"idle"`, `"listening"`, `"hearing"`, `"thinking"` ou `"speaking"` | à chaque changement d'état |
| `text` | texte cumulé de la réponse | pendant la génération (le texte entier à chaque fois : remplacer, ne pas ajouter) |
| `user_text` | transcription de ce que l'utilisateur a dit | après un tour vocal |
| `tool` | nom de l'outil utilisé | une fois par outil et par tour (le texte, lui, ne contient pas cette ligne) |
| `audio-start` | `{id, text, element}` | début d'un bloc de voix : `text` est la phrase dite, `element` le `<audio>` qui la joue |
| `audio-end` | rien | fin de la voix de la réponse |
| `level` | `{source: "mic" ou "voice", value: 0 à 1}` | niveau sonore toutes les 50 ms (voir plus bas) |
| `done` | `{ms}`, `{stopped:true}`, `{empty:true}`, `{ignored:true}` ou `{stopped:true, reason:"keyword"}` | fin du tour |
| `conversation` | `{state: "open" ou "closed"}` | seulement si l'option `conversation` est active : une conversation vient de s'ouvrir ou de se fermer |
| `warning` | `{code, message}` | avertissement sans gravité, par exemple `system_ignored` (voir « Prompt système de l'application ») |
| `error` | `{code, message}` | erreur réseau, quota, clé refusée, modèle… (codes dans la section API) |

Signification des états : `idle` (rien), `listening` (micro ouvert, silence), `hearing` (l'utilisateur parle), `thinking` (Aria réfléchit),
`speaking` (Aria parle). Parler pendant qu'Aria parle baisse sa voix, puis la coupe quand la phrase est terminée et envoyée.

### Conversation à mots-clés (option du profil)

Comme l'interface d'Aria, un agent peut ne répondre qu'à l'intérieur d'une **conversation** ouverte par un mot, et se couper sur un mot d'arrêt. C'est une option du **profil**,
**désactivée par défaut** : sans elle, l'agent répond à toute phrase et `/v1/turn` est exactement celui d'avant.

```json
"profiles": { "robot": { "servers": ["memoire"], "conversation": {
    "enabled": true,
    "open_phrases":  ["bonjour bamboo", "ok bamboo"],
    "close_phrases": ["merci bamboo", "bonne nuit"],
    "stop_phrases":  ["stop", "tais toi", "silence"] } } }
```

| Phrase | Effet |
|---|---|
| **dite** (voix) avec un mot d'ouverture **au début** | ouvre la conversation (événement `conversation {state:"open"}`) et répond ; la suite de la phrase est la question |
| **dite** hors conversation, sans mot d'ouverture | **ignorée en silence** : ni modèle, ni historique, ni journal d'audit ; `done {ignored:true}` |
| **saisie** (texte) | part toujours : le texte ne peut pas ouvrir de conversation, mais peut la fermer |
| avec un mot de **clôture**, pendant la conversation | est répondue, puis la conversation se ferme (`conversation {state:"closed"}`), l'historique est effacé et le journal d'audit change de conversation |
| **courte** et contenant un mot d'**arrêt**, à tout moment | coupe la réponse, sans appeler le modèle : `done {stopped:true, reason:"keyword"}` (l'arrêt marche même si `enabled` est faux, dès que `stop_phrases` est renseigné) |

- L'état est **propre à chaque session** (`session`) : deux utilisateurs d'un même agent ne s'ouvrent pas mutuellement.
- Un tour **audio** compte toujours dans le quota quotidien, même ignoré (la transcription précède la décision).
- Limites : 40 phrases par liste, 60 caractères par phrase ; comparaison sans accents ni majuscules ni ponctuation. Une faute dans l'option refuse la sauvegarde du profil.

### Prompt système de l'application

L'application peut donner ses propres consignes au LLM (nom du site, ton, sujets, contexte de la page) avec l'option `systemPrompt`, **à l'initialisation** :

```js
const aria = AriaClient.create({
  baseUrl: "https://ARIA:7861", apiKey: "ak_…",
  session: "utilisateur-42",
  systemPrompt: "Tu es l'assistante du site Bleu, une boutique de vélos. Tu tutoies. L'utilisateur est sur la page « Entretien »."
});
```
- **Il complète, il ne remplace pas.** Le message système d'une session est celui du profil de la clé (fixé par l'administrateur d'Aria : règles, limites, outils),
  suivi des consignes de l'application sous le titre « Consignes de l'application ». Un prompt qui demande d'ignorer les règles précédentes n'enlève rien : les
  règles du profil restent en tête. Les outils disponibles sont toujours ceux du profil.
- **Il est fixé à la création de la session.** Le client l'envoie à chaque tour, mais le serveur ne le retient qu'au premier tour de la session `(clé, session)`.
  Un prompt différent envoyé ensuite est ignoré et l'événement `warning` (`code: "system_ignored"`) l'indique. Pour changer de consignes, appeler
  `await aria.newSession({ systemPrompt: "nouvelles consignes" })` : l'ancienne session est fermée (elle libère sa place) et la suivante démarre avec un
  historique vide. `reset()`, lui, efface l'historique mais **garde** les consignes.
- **Limites :** 4000 caractères au plus (`413`, code `too_large`, sans consommer de quota), un texte seulement (`400` sinon), caractères de contrôle retirés,
  retours à la ligne conservés. Un prompt vide équivaut à aucun prompt.
- **En appel direct :** champ `system` du corps de `POST /v1/turn` (voir la section API).
- **Ne pas y mettre de secret** : le texte passe par le navigateur de l'utilisateur et n'est pas confidentiel.

### Habiller son propre visuel

**1. En CSS seulement : `bindState(élément)`.** Le client écrit sur l'élément l'attribut `data-aria-state` et la variable CSS `--aria-level`
(0 à 1, niveau sonore de la voix d'Aria quand elle parle, du micro quand l'utilisateur parle). Tout le style reste celui de la page :

```html
<div id="halo"></div>
<style>
  #halo { --aria-level: 0; width: 96px; height: 96px; border-radius: 50%; border: 3px solid #ccc;
          box-shadow: 0 0 calc(8px + var(--aria-level) * 40px) #2a6df4; }
  #halo[data-aria-state="listening"] { border-color: #16a34a; }
  #halo[data-aria-state="thinking"]  { animation: pulse 1s infinite; }
  #halo[data-aria-state="speaking"]  { border-color: #2a6df4; }
</style>
<script> aria.bindState(document.getElementById("halo")); </script>
```
`bindState(el, {level: false})` ne pose que l'état (pas de mesure du niveau, donc aucune analyse audio).

**2. Niveau sonore : l'événement `level`.** Une valeur 0 à 1 toutes les 50 ms, pour une jauge, une animation ou un avatar :
```js
aria.on("level", ({source, value}) => { if (source === "voice") bouche.style.transform = `scaleY(${0.2 + value})`; });
```
`source` vaut `"voice"` (la voix d'Aria, un dernier `0` est envoyé à la fin) ou `"mic"` (celle de l'utilisateur, seulement pendant l'écoute).

**3. Forme d'onde ou spectre : `analyser()`.** Renvoie un `AnalyserNode` standard, à lire dans la boucle d'animation de la page :
```js
const an = aria.analyser("voice");               // "mic" : null tant que l'écoute n'est pas démarrée
const data = new Uint8Array(an.fftSize);
(function dessiner() { an.getByteTimeDomainData(data); /* …dessiner sur un canvas… */ requestAnimationFrame(dessiner); })();
```
`an.getByteFrequencyData(…)` donne le spectre. L'analyseur de la voix est inséré avant les haut-parleurs : le son reste audible. Celui du micro n'est
**jamais** relié aux haut-parleurs (pas d'écho). L'analyse n'est installée que si la page la demande, pour ne rien coûter aux pages qui n'en ont pas besoin.

**4. Suivre le texte dit : `audio-start`.** `text` donne la phrase en cours de lecture (sous-titres, surbrillance) et `element` l'élément `<audio>` du bloc
(durée, `currentTime`, pause).

Un exemple complet (halo CSS, forme d'onde, boutons) est dans `docs/agents-demo.html`.

### Exigences et pièges

- Le **micro exige une page en HTTPS** (ou `localhost`) ; le navigateur doit avoir **accepté une fois le certificat** d'Aria (ouvrir
  `https://ARIA:7861/v1/health` dans un onglet).
- La lecture audio exige un **geste de l'utilisateur** (clic) avant la première réponse vocale : appeler `start()` ou `say()` depuis un gestionnaire de clic.
- Le VAD est chargé depuis jsDelivr (option `cdn` pour l'héberger ailleurs) ; le modèle vient d'Aria.
- Un gestionnaire d'événement qui lève une exception est consigné dans la console mais ne casse jamais le client.
- Un seul tour à la fois par session : un nouveau `say()` ou une nouvelle phrase remplace le tour en cours.
- Si l'écoute ne réagit pas, passer `debug: true` : la console indique la probabilité de parole vue par le VAD. Un niveau constant très bas vient presque
  toujours du micro (coupé, mauvais périphérique) et non d'Aria.

## API (HTTP, `Authorization: Bearer ak_…` ou `X-Api-Key`)

| Route | Rôle |
|---|---|
| `POST /v1/turn` | `{"text": "…"}` ou `{"audio": "data:audio/wav;base64,…", "session": "…", "speak": true, "system": "consignes de l'application"}` ; réponse `text/event-stream` |
| `POST /v1/stop` | coupe la réponse en cours de la session |
| `POST /v1/reset` | efface l'historique de la session |
| `POST /v1/close` | ferme la session `{"session": "…"}` de la clé (jamais celle d'une autre clé) et libère sa place ; sans effet si elle n'existe pas |
| `GET /v1/health` | vérifie la clé |
| `GET /aria-client.js`, `GET /assets/silero_vad.onnx` | fichiers publics du client |

Événements du flux : `state`, `user_text`, `text`, `tool`, `audio` (`{id, format, data, text}`, wav en base64), `warning`, `done`, `error`. Une session est
identifiée par `(clé, session)` : historique et outils indépendants. Codes : 401 clé invalide, 403 origine refusée ou voix interdite, 413 trop gros,
429 débit / quota / blocage, 503 trop de sessions.

## Sécurité

Clés hachées (HMAC-SHA256), comparaison à temps constant, montrées une fois ; révocation immédiate (les sessions ouvertes sont fermées) ; expiration ;
**domaines autorisés par clé** (en-tête `Origin`, CORS limité à l'origine exacte, jamais `*`) ; débit par clé et par IP, quota de tours par jour,
blocage temporaire de l'IP après des échecs répétés ; plafonds de taille (corps 8 Mo, audio 30 s, texte 2000 caractères). L'administration est servie sur
`127.0.0.1` seulement, avec jeton et `Origin` obligatoire. **Toutes les applications consomment votre clé du LLM** : les quotas par clé en bornent le coût.
Une clé sans domaine autorisé refuse tout navigateur ; un client hors navigateur (curl, serveur) n'envoie pas d'`Origin` et n'est limité que par la clé.

## Tests

`node tests/test_aria_client.js` (client, avec un faux Web Audio) ; dans le conteneur : `python -m unittest discover -s tests` après copie du dossier `tests/`
(les tests de l'API nécessitent `starlette`, absent du poste hôte).
