import asyncio
import os
import json
import components.utils as utils
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

class Mcp:
    def __init__(self, mcp_params):
        self.params = mcp_params or {}
        self.servers_conf = self.params.get("servers", {})
        self.tool_registry = {}  
        self.openai_tools = []   
        self.stack = AsyncExitStack()
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
                for tool in tools_resp.tools:
                    self.tool_registry[tool.name] = session
                    self.openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                utils.log_info("MCP", f"✅ Serveur [{name}] prêt : {len(tools_resp.tools)} outils enregistrés.")
            except Exception as e:
                utils.log_info("MCP", f"❌ Échec de connexion au serveur [{name}]: {e}")

    async def get_tools_for_openai(self):
        """Récupère les schémas d'outils pour LM Studio."""
        if not self.tool_registry:
            await self._connect_all()
        return self.openai_tools if self.openai_tools else None

    async def call_tool(self, tool_name, arguments):
        """Version asynchrone (utilisée par LM Studio) avec logs détaillés."""
        if tool_name not in self.tool_registry:
            utils.log_info("MCP", f"⚠️ Outil inconnu appelé : {tool_name}")
            return f"Erreur : L'outil {tool_name} n'existe pas."
        
        # LOG : Entrée
        utils.log_info("MCP", f"🚀 APPEL OUTIL : [{tool_name}] | Paramètres : {json.dumps(arguments)}")
        
        session = self.tool_registry[tool_name]
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            
            # Extraction de la réponse
            response_text = result.content[0].text if result.content else "Aucune réponse"
            
            # LOG : Sortie
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