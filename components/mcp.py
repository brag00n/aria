import asyncio
import os
import json
import components.utils as utils
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

# Les outils dont le nom commence ainsi sont réservés à l'interface d'Aria : appelables par elle, absents des schémas du LLM
HIDDEN_PREFIX = "ui_"


class Mcp:
    def __init__(self, mcp_params, secret_for_servers="", hosted_only=()):
        self.params = mcp_params or {}
        self.hosted_only = frozenset(hosted_only or ())          # serveurs démarrés pour les accès tiers : leurs outils ne sont PAS ceux de l'interface
        self.secret_for_servers = secret_for_servers          # secret du canal HTTP de la mémoire (processus d'Aria seulement ; vide pour les tiers)
        self.servers_conf = self.params.get("servers", {})
        self.tool_registry = {}  
        self.openai_tools = []   
        self.stack = AsyncExitStack()
        self._lock = None            # créé à la première utilisation, dans la boucle qui s'en sert
        self.loop = asyncio.get_event_loop()
        
    def start(self):
        """Démarrage initial (mode local)."""
        utils.log_info("MCP", "Démarrage des services MCP...")
        try:
            if self.loop.is_running():
                asyncio.create_task(self._connect_all())
            else:
                self.loop.run_until_complete(self._connect_all())
        except Exception as e:
            utils.log_info("MCP", f"❌ Erreur critique au démarrage: {e}")

    async def _connect_all(self):
        for name, conf in self.servers_conf.items():
            try:
                conf = utils.expand_secrets(conf, f"MCP {name}")   # tags <SECRET> -> valeurs du .env
                hidden_tools = set(conf.get("hide_tools") or ())   # outils de ce serveur jamais proposés au LLM (ex. identifier_interlocuteur pour un tiers)
                if self.secret_for_servers:
                    from components.access import memory_link
                    conf = memory_link.with_secret(conf, self.secret_for_servers)   # serveur mémoire d'Aria : secret de son canal HTTP, en mémoire seulement
                if conf.get("url"):
                    # Serveur distant (HTTP) : "url" et, au besoin, "headers" (ex. Authorization)
                    read, write, _ = await self.stack.enter_async_context(
                        streamablehttp_client(conf["url"], headers=conf.get("headers") or None))
                else:
                    server_params = StdioServerParameters(
                        command=conf.get("run", "uv"),
                        args=conf.get("args", []),
                        env={**os.environ, **conf.get("env", {})}
                    )
                    read, write = await self.stack.enter_async_context(stdio_client(server_params))
                session = await self.stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                tools_resp = await session.list_tools()
                if name in self.hosted_only:
                    # Serveur réservé aux accès tiers (mémoire d'un tiers hors foyer) : il tourne, mais ses outils ne sont jamais enregistrés pour
                    # l'interface d'Aria, sinon ils écraseraient ceux du serveur « memoire » (mêmes noms d'outils).
                    utils.log_info("MCP", f"✅ Serveur [{name}] démarré pour les accès tiers (outils non exposés à l'interface).")
                    continue
                for tool in tools_resp.tools:
                    self.tool_registry[tool.name] = session
                    if tool.name.startswith(HIDDEN_PREFIX) or tool.name in hidden_tools:
                        continue            # appelable par Aria (call_tool), jamais proposé au LLM
                    self.openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                hidden = sum(1 for t in tools_resp.tools if t.name.startswith(HIDDEN_PREFIX))
                shown = len(tools_resp.tools) - hidden
                utils.log_info("MCP", f"✅ Serveur [{name}] prêt : {shown} outils enregistrés"
                                      + (f" (+{hidden} réservé(s) à l'interface)." if hidden else "."))
            except Exception as e:
                utils.log_info("MCP", f"❌ Échec de connexion au serveur [{name}]: {e}")

    async def ensure_connected(self):
        """Connecte les serveurs une seule fois, même si le démarrage d'Aria et la première question se croisent."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if not self.tool_registry:
                await self._connect_all()

    async def close(self):
        """Ferme les sessions MCP. À appeler depuis la tâche qui les a ouvertes (contrainte de la bibliothèque asynchrone)."""
        try:
            await self.stack.aclose()
        except Exception as e:
            utils.log_info("MCP", f"Fermeture des services MCP : {type(e).__name__}: {e}")

    def has_tool(self, name):
        """Un serveur MCP fournit-il cet outil ? (Sert à masquer une fonction de l'interface quand son serveur est absent.)"""
        return name in self.tool_registry

    async def get_tools_for_openai(self):
        """Récupère les schémas d'outils pour LM Studio."""
        await self.ensure_connected()
        return self.openai_tools if self.openai_tools else None

    async def call_tool(self, tool_name, arguments):
        """Version asynchrone (utilisée par LM Studio) avec logs détaillés."""
        if tool_name not in self.tool_registry:
            utils.log_info("MCP", f"⚠️ Outil inconnu appelé : {tool_name}")
            return f"Erreur : L'outil {tool_name} n'existe pas."
        
        # Les lectures d'état de l'interface (ui_etat avec activer='?') ont lieu après chaque message : on ne les trace pas.
        # Un changement d'état et toute erreur restent tracés (et le serveur de mémoire journalise les changements d'interlocuteur).
        quiet = tool_name.startswith(HIDDEN_PREFIX) and (arguments or {}).get("activer", "?") == "?"

        # LOG : Entrée
        if not quiet:
            utils.log_info("MCP", f"🚀 APPEL OUTIL : [{tool_name}] | Paramètres : {json.dumps(arguments)}")
        
        session = self.tool_registry[tool_name]
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            
            # Extraction de la réponse
            response_text = result.content[0].text if result.content else "Aucune réponse"
            
            # LOG : Sortie
            if not quiet:
                utils.log_info("MCP", f"📥 RETOUR OUTIL : [{tool_name}] | Réponse : {response_text[:150]}...")
            
            return response_text
        except Exception as e:
            utils.log_info("MCP", f"❌ ERREUR OUTIL : [{tool_name}] | Message : {str(e)}")
            return f"Erreur lors de l'exécution de {tool_name}: {str(e)}"

    def get_tools_schema(self):
        """Pour compatibilité avec le moteur local."""
        return self.openai_tools if self.openai_tools else None

    def call_tool_sync(self, tool_name, arguments):
        """Version synchrone (utilisée par le mode local) avec logs."""
        if tool_name not in self.tool_registry:
            return "Outil introuvable."
        
        utils.log_info("MCP", f"🚀 (Sync) APPEL OUTIL : [{tool_name}]")
        session = self.tool_registry[tool_name]
        try:
            result = self.loop.run_until_complete(session.call_tool(tool_name, arguments))
            res_text = result.content[0].text
            utils.log_info("MCP", f"📥 (Sync) RETOUR OUTIL : [{tool_name}] | {res_text[:100]}...")
            return res_text
        except Exception as e:
            utils.log_info("MCP", f"❌ (Sync) ERREUR : {e}")
            return str(e)