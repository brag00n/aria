# Mémoire à long terme d'Aria

Serveur MCP (stdio) autonome : il n'importe rien depuis Aria et n'utilise que la bibliothèque standard, plus le paquet `mcp`
déjà présent dans l'image. Les données sont **100 % locales**, en JSONL lisible.

## Outils exposés (5 au LLM + 1 réservé à l'interface)

| Outil | Rôle |
|---|---|
| `memoriser(fait, type, tags)` | retient une information de l'interlocuteur actif. `type` : `fait`, `preference`, `tache`, `ephemere` (privés) ou `foyer` (partagé). Refusé si l'interlocuteur est inconnu |
| `rechercher_souvenir(question, limite)` | retrouve les souvenirs de l'interlocuteur actif et ceux du foyer ; question vide = les plus importants |
| `oublier(id)` | oublie un souvenir (il reste dans l'archive) |
| `identifier_interlocuteur(nom, complement, surnom)` | désigne (ou crée) la personne qui parle ; signale les homonymes |
| `relier(personne_a, lien, personne_b, precision)` | mémorise « a est le `lien` de b » (« moi » = l'interlocuteur actif) ; `lien` est pris dans une **liste fermée** : `conjoint`, `parent`, `enfant`, `fratrie`, `demi`, `beauparent`, `beauenfant`, `ami`, `collegue`, `voisin`, ou `autre` + `precision` (ex. marraine de). Hors liste : refus avec la liste. Le sexe vient de la fiche de la personne, pas du lien |
| `ui_etat(activer)` | **caché au LLM** (préfixe `ui_`) : interlocuteurs, actif et consigne du tour, pour l'interface d'Aria |

## Interlocuteurs

- L'**interlocuteur actif** vit dans le serveur (jamais dans les arguments d'un outil) : le LLM ne peut pas mémoriser « en tant que » quelqu'un d'autre. Au démarrage il est **Inconnue** ; Aria le resynchronise à chaque message avec la liste déroulante.
- **Inconnue = rien n'est mémorisé.** Aria reçoit alors la consigne de demander à qui elle parle, puis d'appeler `identifier_interlocuteur`.
- Chaque souvenir a un propriétaire `o` (id de personne). **Vide = partagé** : types `foyer` et `relation`, et tous les souvenirs créés avant les interlocuteurs. Chacun voit **les siens et le partagé**.
- Les personnes sont dans `personnes.jsonl` (`id`, `nom`, `compl` = nom de famille, `surnom`). Deux « Olivier » se distinguent par `compl` : la liste affiche « Olivier (Martin) ».
- Les **relations** (`t="relation"`, champ `rel` : `a`, `l`, `b`, `ia`, `ib`) sont partagées et ne décroissent jamais (tag `core`).
- Rattacher un ancien souvenir, fusionner deux personnes, créer ou renommer : **page de gestion**, onglets Souvenirs, Personnes et Relations.

## Accès tiers : des personnes de type système

Il n'y a **qu'un seul serveur de mémoire** : c'est lui qui garantit l'intégrité et la confidentialité. Une application ou un appareil tiers (un robot) y est une
**personne de type système** (`type: "systeme"`, `cle` = identifiant interne de son accès, voir `docs/access.md`). Absence de `type` = humain : les fichiers existants
se lisent sans migration. L'identité suit la **clé interne** : renommer l'accès ou régénérer sa clé API ne fait pas perdre sa mémoire.

| | Système (accès tiers) | Humain (Aria, page de gestion) |
|---|---|---|
| Personnes et relations (graphe, parenté) | **toutes**, en lecture | toutes |
| Souvenirs | **les siens** + le partagé de type `foyer` ; jamais les souvenirs privés d'un humain ni les anciens souvenirs partagés | les siens + le partagé |
| Ce qu'il écrit | toujours **privé** (propriétaire = lui), même de type `foyer` ou `relation` | comme avant |
| Liens (`relier`) | seulement entre **lui** (« moi ») et une personne déjà connue : il ne crée ni lien entre deux humains ni nouvelle personne | tous |
| Oublier | ses souvenirs seulement | les siens et le partagé |
| Changer d'identité (`identifier_interlocuteur`, `ui_*`) | **impossible**, refusé par le serveur | oui |
| Débit d'écriture | borné **par principal** (un tiers bavard ne bloque pas Aria) | idem |

- Les systèmes n'apparaissent pas dans la liste déroulante d'Aria (ce ne sont pas des interlocuteurs) ; la page de gestion les montre avec le badge « ⚙ système » et
  les souvenirs de chacun, et refuse de fusionner un système avec une personne.
- **Le serveur impose l'identité**, jamais le client : retirer un outil du schéma du LLM n'est qu'un confort.

### Deux façades, un seul processus

`server.py` sert **Aria en stdio** (comportement inchangé : l'interlocuteur actif, choisi dans la liste déroulante) et, si `MEMORY_HTTP_PORT` est défini, un **canal
HTTP** (`http_facade.py`) pour les accès tiers. Les deux appellent le même cœur (`service.py`), donc le même `Store`, le même verrou et le même écrivain.

Le canal HTTP : écoute sur **127.0.0.1 seulement** ; chaque requête doit porter `Authorization: Bearer <secret>` (comparé à temps constant) — `MEMORY_HTTP_SECRET`, d'au moins
16 caractères, généré par Aria à chaque démarrage et transmis par l'environnement du processus (jamais écrit sur disque) ; **sans secret valide, le canal ne s'ouvre pas**.
L'identité vient de `X-Memory-Principal` (clé interne de l'accès) et `X-Memory-Name` (nom, encodé en pourcentage). Seuls `memoriser`, `rechercher_souvenir`, `oublier` et
`relier` sont servis.

## Importance (calculée par le serveur, pas par le LLM)

- Note de départ selon le type : préférence 6, fait 5, tâche 4, éphémère 2.
- Renfort `s` (1,0 à 2,0) : +0,1 à chaque rappel, +0,25 si le même fait est redit.
- `eff = imp × s × 0,5^(âge / demi-vie)`. Demi-vies : éphémère 10 j, tâche 30 j, préférence 90 j, fait 120 j.
- `eff < 1` : masqué des recherches. `eff < 0,5` depuis plus de 30 jours : archivé à la compaction (jamais supprimé).
- Tags `pinned`, `core`, `identity` : aucune décroissance.

## Fichiers (`$MEMORY_DIR`, par défaut `models/memory/`, ignoré par git)

```
default/personnes.jsonl    personnes connues (interlocuteurs)
default/memories.jsonl     instantané (un souvenir par ligne), écrit seulement par la compaction
default/events.jsonl       journal append-only depuis l'instantané
default/archive/AAAA-MM.jsonl   oubliés et périmés : {"why","ts","r":{...souvenir...}}
default/backups/*.jsonl.gz      7 dernières sauvegardes de l'instantané
logs/activity.jsonl        journal d'activité (rotation à 5 Mo, 5 fichiers)
```

Exemples de lignes :
```
{"id":"m_6edcbd","t":"fait","txt":"La chatte s'appelle Réglisse","imp":5.0,"tags":["animal"],"h":"…","c":"2026-10-30T13:10:21+00:00","last":"…","hits":0,"s":1.0,"src":"llm"}
{"op":"add","ts":"…","r":{…souvenir complet…}}      {"op":"touch","ts":"…","id":"m_6edcbd"}
{"op":"update","ts":"…","id":"m_6edcbd","set":{"tags":["pinned"]}}      {"op":"del","ts":"…","id":"m_6edcbd","why":"user"}
```
`src` : `llm` (outil MCP), `ui` (page de gestion) ou `main` (ajouté à la main). Dates en UTC. `o` : propriétaire (vide = partagé).
Les onglets Souvenirs, Personnes et Relations ont chacun une barre de recherche (sans accents ni majuscules, tous les mots doivent correspondre). Personnes et Relations y ajoutent des **critères en listes déroulantes** : pour les personnes, le champ cherché, le sexe, le statut (connue par Aria ou seulement citée) et la présence de souvenirs ; pour les relations, le champ cherché, la personne (des deux côtés), le lien et le type (famille ou connaissance). L'onglet Relations a aussi une **barre de création** (personne, lien de la liste fermée, personne, précision pour « autre »), route `POST /api/relation/add` : les deux personnes sont désignées par leur fiche, le lien est refusé s'il existe déjà (même dans l'autre sens). Une relation ne se crée plus comme un souvenir libre. Dans la fiche d'une personne, la case **Connue par Aria** l'ajoute à la liste des interlocuteurs (comme si elle avait déjà discuté) ou l'en retire (elle redevient « citée » : ses liens et ses souvenirs restent) ; l'interlocuteur actif ne peut pas être retiré. Une personne citée qui prend la parole devient connue d'elle-même. Les extrémités d'une relation sont retrouvées par l'**identifiant** de leur fiche (`ia`, `ib`), jamais par le nom : renommer une personne dans la page (ou fusionner deux personnes) ne casse aucun lien, et le texte du souvenir est réécrit avec le nouveau nom (`synchroniser_noms`). Au démarrage, `migrer()` donne un type (`t`) aux relations plus anciennes, déduit de leur texte (« marié à » => `conjoint`, ce qui n'est pas reconnu => `autre`), complète leurs identifiants de personnes et ne touche ni au texte ni au lien d'origine ; elle est idempotente. Ligne de relation : `{"id":"m_b2d7bb","t":"relation","txt":"Tiphaine est la femme de Olivier","o":"","rel":{"a":"Tiphaine","l":"femme de","b":"Olivier","t":"conjoint","ia":"p2","ib":"p1"},"tags":["core"],…}`.
Personne : `{"id":"p1","nom":"Olivier","compl":"Cousinier","surnom":"Oli","c":"2026-10-30T18:42:26+00:00"}`.

## Consulter et modifier à la main

```
grep -i réglisse models/memory/default/memories.jsonl models/memory/default/events.jsonl
python tools/memory/analyze.py models/memory          # résumé du journal d'activité (--json pour une sortie machine)
```
Le serveur détecte une modification externe (taille ou date) et recharge à l'appel suivant. Pour éditer **à la main** sans risque :
arrêter Aria, modifier `memories.jsonl` (un objet JSON par ligne, LF), supprimer `events.jsonl`, relancer. Une ligne illisible est
ignorée et signalée, jamais fatale ; une dernière ligne interrompue par un arrêt brutal est retirée au démarrage.
Restaurer une sauvegarde : `gunzip -c backups/memories-XXXX.jsonl.gz > memories.jsonl` puis supprimer `events.jsonl`.

## Page de gestion (optionnelle)

Variable `MEMORY_UI_PORT` (ex. 7870) dans l'entrée `memoire` de `configs/default.json`. Sans elle, aucune page. La page liste,
cherche, ajoute, modifie, **attribue à une personne**, protège (📌), oublie et restaure les souvenirs, gère les **personnes** (création,
renommage, fusion) et les **relations**, et affiche archive, statistiques et journal. Elle voit tous les souvenirs, quel que soit l'interlocuteur actif. Pour l'atteindre
depuis le PC hôte, publier le port dans `docker-compose.yml` **sur 127.0.0.1 uniquement** : `"127.0.0.1:7870:7870"`.
Le bouton 🕸 d'une personne (onglet Personnes) ouvre un **graphe de ses relations immédiates** (route `GET /api/graphe?id=`) : la personne au centre, ses proches autour, en bleu la famille (conjoint, parents, enfants, fratrie), en orange les connaissances (amis, voisins, collègues). Un clic sur un proche recentre le graphe sur lui. Un sélecteur règle le niveau (1 par défaut, 5 au maximum, `&niveau=` dans la route) : le niveau 2 ajoute les proches des proches, et ainsi de suite.
Autres variables : `MEMORY_UI_TOKEN` (jeton, à passer en `?token=`), `MEMORY_UI_HOSTS` (noms d'hôte supplémentaires autorisés).

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `MEMORY_DIR` | `<repo>/models/memory` | dossier des données |
| `MEMORY_NAMESPACE` | `default` | sous-dossier (un espace de noms par utilisateur, plus tard) |
| `MEMORY_UI_PORT` | absent | port de la page de gestion |
| `MEMORY_HTTP_PORT` | absent | port du canal HTTP des accès tiers (127.0.0.1 seulement ; absent = pas de canal) |
| `MEMORY_HTTP_SECRET` | absent | secret exigé par ce canal (≥ 16 caractères) ; sans lui, le canal n'est pas ouvert. Fourni par Aria, à ne pas écrire dans la config |
| `MEMORY_LOG_TEXT` | `1` | `0` : le journal d'activité garde une empreinte à la place du texte des requêtes |
| `MEMORY_LOG_LEVEL` | `info` | `debug` ajoute les scores de recherche |

## Vie privée

Les souvenirs sont en clair sur le disque (droits par défaut du système). Le journal d'activité ne recopie pas le texte des
souvenirs, seulement leurs identifiants, mais il contient les **requêtes de recherche**. Aucun appel réseau.

## Tests

```
python -m unittest discover -s tools/memory/tests         # tests unitaires, page, analyse
MEMORY_BENCH=1 python -m unittest discover -s tools/memory/tests   # + benchmark 10 000 souvenirs
python -m unittest tools.memory.tests.test_e2e_stdio      # bout en bout par le protocole MCP (nécessite le paquet mcp : dans le conteneur)
python -m unittest tools.memory.tests.test_http_facade    # canal HTTP des accès tiers : secret, identité imposée, outils servis (conteneur)
python -m unittest tools.memory.tests.test_principaux     # personnes système : isolation des souvenirs, graphe commun, usurpation
```

## Supprimer la fonctionnalité

1. Supprimer le dossier `tools/memory/`.
2. Retirer l'entrée `memoire` de `mcp.params.servers` dans `configs/default.json`, et la règle « Mémoire » des `system_message`.
3. Retirer la ligne de port 7870 de `docker-compose.yml` si elle a été ajoutée.
4. Éventuellement effacer `models/memory/` (données et journaux).

Aucun changement du code d'Aria ni des dépendances.
