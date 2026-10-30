# Mémoire à long terme d'Aria

Serveur MCP (stdio) autonome : il n'importe rien depuis Aria et n'utilise que la bibliothèque standard, plus le paquet `mcp`
déjà présent dans l'image. Les données sont **100 % locales**, en JSONL lisible.

## Outils exposés (3)

| Outil | Rôle |
|---|---|
| `memoriser(fait, type, tags)` | retient une information durable. `type` : `fait`, `preference`, `tache`, `ephemere` |
| `rechercher_souvenir(question, limite)` | retrouve des souvenirs ; question vide = les plus importants |
| `oublier(id)` | oublie un souvenir (il reste dans l'archive) |

## Importance (calculée par le serveur, pas par le LLM)

- Note de départ selon le type : préférence 6, fait 5, tâche 4, éphémère 2.
- Renfort `s` (1,0 à 2,0) : +0,1 à chaque rappel, +0,25 si le même fait est redit.
- `eff = imp × s × 0,5^(âge / demi-vie)`. Demi-vies : éphémère 10 j, tâche 30 j, préférence 90 j, fait 120 j.
- `eff < 1` : masqué des recherches. `eff < 0,5` depuis plus de 30 jours : archivé à la compaction (jamais supprimé).
- Tags `pinned`, `core`, `identity` : aucune décroissance.

## Fichiers (`$MEMORY_DIR`, par défaut `models/memory/`, ignoré par git)

```
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
`src` : `llm` (outil MCP), `ui` (page de gestion) ou `main` (ajouté à la main). Dates en UTC.

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
cherche, ajoute, modifie, protège (📌), oublie et restaure les souvenirs, et affiche statistiques et journal. Pour l'atteindre
depuis le PC hôte, publier le port dans `docker-compose.yml` **sur 127.0.0.1 uniquement** : `"127.0.0.1:7870:7870"`.
Autres variables : `MEMORY_UI_TOKEN` (jeton, à passer en `?token=`), `MEMORY_UI_HOSTS` (noms d'hôte supplémentaires autorisés).

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `MEMORY_DIR` | `<repo>/models/memory` | dossier des données |
| `MEMORY_NAMESPACE` | `default` | sous-dossier (un espace de noms par utilisateur, plus tard) |
| `MEMORY_UI_PORT` | absent | port de la page de gestion |
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
```

## Supprimer la fonctionnalité

1. Supprimer le dossier `tools/memory/`.
2. Retirer l'entrée `memoire` de `mcp.params.servers` dans `configs/default.json`, et la règle « Mémoire » des `system_message`.
3. Retirer la ligne de port 7870 de `docker-compose.yml` si elle a été ajoutée.
4. Éventuellement effacer `models/memory/` (données et journaux).

Aucun changement du code d'Aria ni des dépendances.
