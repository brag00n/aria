"""Cœur des outils de mémoire : les mêmes fonctions servent la façade stdio (Aria) et la façade HTTP (accès tiers).

Chaque fonction prend un PRINCIPAL : l'identifiant de la personne qui parle. Pour Aria (stdio), c'est l'interlocuteur actif (`store.active`, changé par
la liste déroulante) ; pour un accès tiers (HTTP), c'est la personne SYSTÈME de l'accès, imposée par le serveur d'après l'en-tête authentifié,
jamais par le LLM. Les règles de visibilité sont dans `Store.visible_to` : un système ne lit que ses souvenirs (et le foyer partagé), mais connaît toutes
les personnes et leurs liens.
"""


def _line(rec):
    """Une ligne compacte par souvenir, c'est tout ce que le LLM reçoit."""
    return f"{rec['id']} [{rec['t']}] {rec['txt']}"


class Service:
    def __init__(self, store, interlocuteurs):
        self.store = store
        self.i = interlocuteurs

    # ------------------------------------------------------------------ outils du LLM
    def identifier(self, nom, complement="", surnom="", principal=None, trace=None):
        return self.i.identifier(nom, complement, surnom, principal=principal, trace=trace)

    def relier(self, personne_a, lien, personne_b, precision="", principal=None, trace=None):
        return self.i.relier(personne_a, lien, personne_b, precision, principal=principal, trace=trace)

    def memoriser(self, fait, type="fait", tags="", principal=None, trace=None):
        status, rec = self.store.add(fait, type, tags, src="llm", owner=principal, trace=trace)
        if status == "added":
            return f"OK {rec['id']}"
        if status == "duplicate":
            return f"Déjà connu ({rec['id']}), renforcé."
        return self.i.refus(status)

    def rechercher(self, question="", limite=5, principal=None):
        results, fallback = self.store.search(question, limite, src="llm", owner=principal)
        # Le graphe du foyer (personnes et liens) est commun : un système le lit comme un humain. Seuls les souvenirs sont filtrés.
        proches = self.i.entourage(question, deja={r["id"] for r in results}, principal=principal)
        calcules = self.i.liens_calcules(question, principal=principal)
        if not results and not proches and not calcules:
            return "Aucun souvenir."
        lines = []
        if calcules:
            lines.append("Parenté établie (à utiliser telle quelle) :")
            lines.extend(f"- {phrase}" for phrase in calcules)
        if fallback and results:
            lines.append("Aucune correspondance exacte, souvenirs principaux :")
        lines.extend(_line(r) for r in results)
        if proches:
            lines.append("Liens connus dans la famille :")
            lines.extend(f"- {r['txt']}" for r in proches)
        return "\n".join(lines)

    def oublier(self, rid, principal=None, trace=None):
        return "Oublié." if self.store.delete((rid or "").strip(), why="user", src="llm", owner=principal, trace=trace) else "Souvenir introuvable."
