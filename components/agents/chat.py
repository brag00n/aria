"""Chat de l'interface d'Aria parlant à l'agent par défaut `aria`, EN INTERNE (même processus, aucune clé, aucun HTTP). Sans Gradio : `app.py` n'y branche que ses composants.

Ce module regroupe ce qui était dispersé dans `app.py` : l'interlocuteur global (liste déroulante), la conversation à mots-clés, l'arrêt, l'oubli du contexte et le journal
d'audit. Le tour lui-même est `turn.run_turn`, le même cœur que l'API : une règle ne se corrige qu'à un endroit.

Un tour du chat renvoie des événements `(nom, données)` de plus haut niveau que ceux du cœur : `texte`, `audio` (chemin du wav : le chat le lit lui-même), `ctl` (ordre pour le
navigateur : stop, idle, conv_open, conv_close), `erreur`. Le tour s'exécute dans la boucle de l'interface (celle du moteur et du MCP globaux), sans pont.
"""
import asyncio

import components.utils as utils
from components import history as history_mod
from components import interlocuteur as interlocuteur_mod
from components.conversation import Conversation

from . import turn

AGENT_ID = "aria"
SESSION = "ui"                      # session de l'agent `aria` utilisée par le chat de l'interface : sid « aria:ui »


class ChatAria:
    def __init__(self, agents, config, mcp_manager, tts=None):
        """`agents` : objet `server.Agents` (registre des sessions, clés). `config` : configuration VIVANTE d'Aria. `mcp_manager` : fonction sans argument qui renvoie le gestionnaire
        MCP global (ou None). `tts` : synthèse vocale de l'interface (None = texte seul)."""
        self.agents, self.config, self.mcp_manager, self.tts = agents, config, mcp_manager, tts
        self.dernier_interlocuteur = ""                               # dernier interlocuteur connu de l'interface (choisi à la main OU fixé par Aria)

    # ------------------------------------------------------------------ session
    @property
    def registre(self):
        return self.agents.agents_deps.sessions

    @property
    def conv(self):
        """`Conversation` de la session du chat (mots d'ouverture, de clôture et d'arrêt), ou None si elle n'existe pas encore."""
        s = self.registre.sessions.get(f"{AGENT_ID}:{SESSION}")
        return s.conv if s else None

    async def session(self):
        rec = self.agents.agents_deps.keys.record(AGENT_ID)
        return await self.registre.get(rec, SESSION)

    def enregistrement(self):
        return self.agents.agents_deps.keys.record(AGENT_ID)

    # ------------------------------------------------------------------ interlocuteur global (liste déroulante)
    async def etat_interlocuteur(self, choix="?"):
        return await interlocuteur_mod.etat(self.mcp_manager(), choix)

    async def reinitialiser_interlocuteur(self):
        """Clôture / nouvelle conversation : Aria ne sait plus à qui elle parlait (retour à « Inconnue »)."""
        await interlocuteur_mod.etat(self.mcp_manager(), "")

    async def rafraichir_interlocuteur(self):
        etat = await self.etat_interlocuteur("?")
        self.dernier_interlocuteur = etat["actif"]                      # suit aussi les changements décidés par Aria (outil identifier_interlocuteur)
        return etat, interlocuteur_mod.disponible(self.mcp_manager())

    async def changer_interlocuteur(self, choix):
        """Changement à la main dans la liste. Passer d'une personne connue à une autre efface l'historique du LLM (pas de confidences de la précédente)."""
        choix = choix or ""
        precedent, self.dernier_interlocuteur = self.dernier_interlocuteur, choix
        await self.etat_interlocuteur(choix)
        if precedent and choix and precedent != choix:
            await self.oublier(force=True)
            utils.log_info("INTERLOCUTEUR", f"Changement {precedent} -> {choix} : historique du LLM effacé")
            return True
        return False

    # ------------------------------------------------------------------ contexte et conversation
    async def oublier(self, force=False):
        """Efface l'historique envoyé au LLM (pas le chat affiché) et change de conversation pour le journal d'audit. Automatique à la clôture si
        `memory.reset_on_close`, sinon seulement si `force` (bouton « Nouvelle conversation », changement d'interlocuteur)."""
        if not (force or history_mod.settings(self.config.get("memory"))["reset_on_close"]):
            return False
        s = await self.session()
        self.registre.reset(s)
        return True

    def arreter(self):
        """Bouton « Stop » ou mot d'arrêt : coupe la génération en cours."""
        s = self.registre.sessions.get(f"{AGENT_ID}:{SESSION}")
        if s is not None:
            s.stop.set()

    async def nouvelle_conversation(self):
        """Bouton « Nouvelle conversation » : Aria oublie tout, la conversation à mots-clés se referme, l'interlocuteur est oublié."""
        self.arreter()
        await self.oublier(force=True)
        c = self.conv
        if c is not None and c.active:
            c.end()
        await self.reinitialiser_interlocuteur()

    # ------------------------------------------------------------------ un tour
    async def tour(self, texte=None, audio=None, interlocuteur_choisi="", speak=True):
        """Un tour du chat. Générateur d'événements `(nom, données)` :
            ("ctl", "conv_open"|"conv_close"|"stop"|"idle")   ordre pour le navigateur
            ("user_text", texte)                              ce que le STT a compris (voix)
            ("texte", texte cumulé)                           réponse en cours
            ("audio", {"path", "text"})                       bloc de voix : le chat lit et supprime le fichier
            ("outil", nom)                                    outil utilisé
            ("erreur", message)                               le tour a échoué
        L'interlocuteur choisi dans la liste fait référence : il est communiqué au serveur de mémoire AVANT le tour, qui renvoie la consigne du tour."""
        s = await self.session()
        rec = self.enregistrement()
        etat = await self.etat_interlocuteur(interlocuteur_choisi or "")
        qui = next((label for label, pid in etat.get("choices", []) if pid == etat.get("actif")), "") or "Inconnue"
        action_close = False
        async for nom, d in turn.run_turn(self.agents.agents_deps, s, rec, texte, audio, speak and self.tts is not None, lambda *a, **k: None,
                                          contexte=etat.get("contexte", ""), qui=qui):
            if nom == "user_text":
                yield "user_text", d["text"]
            elif nom == "conversation":
                if d["state"] == "open":
                    yield "ctl", "conv_open"
                else:
                    action_close = True
            elif nom == "text":
                yield "texte", d["text"]
            elif nom == "audio":
                yield "audio", d
            elif nom == "tool":
                yield "outil", d["name"]
            elif nom == "done":
                if d.get("stopped"):
                    yield "ctl", "stop"
                elif d.get("ignored") or d.get("empty"):
                    yield "ctl", "idle"
            elif nom == "error":
                yield "erreur", d.get("message", "erreur")
        if action_close:
            # Le cœur a DÉJÀ effacé l'historique (selon `memory.reset_on_close`) et changé de conversation pour le journal : on n'y touche pas une seconde fois.
            await self.reinitialiser_interlocuteur()                    # clôture : Aria ne sait plus à qui elle parlait
            yield "ctl", "conv_close"
