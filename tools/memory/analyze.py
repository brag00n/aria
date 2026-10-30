"""Résumé du journal d'activité de la mémoire (bibliothèque standard).

Usage : python tools/memory/analyze.py [dossier_des_données] [--json]
Le dossier par défaut est MEMORY_DIR, sinon <repo>/models/memory. Lit activity.jsonl et ses rotations.
Répond à : la mémoire est-elle utilisée ? les recherches aboutissent-elles ? quels souvenirs ne servent jamais ?
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path


def load_events(log_dir):
    files = sorted(Path(log_dir).glob("activity*.jsonl"), key=lambda p: (p.stem != "activity", p.name), reverse=True)
    for path in files:
        try:
            with open(path, "rb") as f:
                for raw in f:
                    try:
                        yield json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
        except OSError:
            continue


def load_memories(data_dir, namespace="default"):
    """Souvenirs actuels : instantané puis journal (même logique que le serveur, en lecture seule et simplifiée)."""
    base = Path(data_dir) / namespace
    records = {}
    for name in ("memories.jsonl", "events.jsonl"):
        try:
            with open(base / name, "rb") as f:
                for raw in f:
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except ValueError:
                        continue
                    if name == "memories.jsonl":
                        records[obj.get("id")] = obj
                    elif obj.get("op") == "add":
                        records[(obj.get("r") or {}).get("id")] = obj["r"]
                    elif obj.get("op") == "del":
                        records.pop(obj.get("id"), None)
        except OSError:
            continue
    return records


def summarize(data_dir, namespace="default"):
    events = list(load_events(Path(data_dir) / "logs"))
    kinds = Counter(e.get("ev") for e in events)
    searches = [e for e in events if e.get("ev") == "search" and e.get("src") == "llm"]
    adds = [e for e in events if e.get("ev") == "add"]
    # Un repli (aucune correspondance, souvenirs principaux proposés par défaut) n'est pas un vrai rappel
    recalled = Counter(i for e in searches if not e.get("fallback") for i in e.get("ids", []))
    memories = load_memories(data_dir, namespace)
    ms = sorted(e["ms"] for e in searches if "ms" in e)
    per_day = Counter((e.get("ts") or "")[:10] for e in events if e.get("ev") in ("add", "search"))
    return {
        "evenements": dict(kinds),
        "souvenirs": len(memories),
        "recherches_llm": len(searches),
        "recherches_sans_resultat": [e.get("q") for e in searches if not e.get("n_res")],
        "recherches_en_repli": [e.get("q") for e in searches if e.get("fallback")],
        "ajouts": len(adds),
        "doublons": sum(1 for e in adds if e.get("dup")),
        "refus_limite_debit": kinds.get("rate_limit", 0),
        "plus_rappeles": recalled.most_common(5),
        "jamais_rappeles": [i for i in memories if i not in recalled],
        "recherche_ms_p95": ms[int(len(ms) * 0.95) - 1] if len(ms) >= 2 else (ms[0] if ms else None),
        "activite_par_jour": dict(sorted(per_day.items())),
        "erreurs": [e for e in events if e.get("ev") == "error"][-5:],
    }


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    data_dir = Path(args[0]) if args else Path(os.environ.get("MEMORY_DIR") or Path(__file__).resolve().parent.parent.parent / "models" / "memory")
    summary = summarize(data_dir)
    if "--json" in argv:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"Mémoire : {data_dir}")
    print(f"  souvenirs actuels : {summary['souvenirs']}  |  ajouts : {summary['ajouts']} (doublons : {summary['doublons']})")
    print(f"  recherches du LLM : {summary['recherches_llm']}  |  p95 : {summary['recherche_ms_p95']} ms")
    for key, label in (("recherches_sans_resultat", "recherches sans résultat"), ("recherches_en_repli", "recherches en repli")):
        if summary[key]:
            print(f"  {label} : " + " | ".join(str(q) for q in summary[key][:8]))
    if summary["plus_rappeles"]:
        print("  plus rappelés : " + ", ".join(f"{i} ({n}x)" for i, n in summary["plus_rappeles"]))
    if summary["jamais_rappeles"]:
        print(f"  jamais rappelés : {len(summary['jamais_rappeles'])}")
    if summary["refus_limite_debit"]:
        print(f"  refus par la limite d'écritures : {summary['refus_limite_debit']}")
    for e in summary["erreurs"]:
        print(f"  ERREUR {e.get('ts')} {e.get('where', '')} {e.get('msg', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
