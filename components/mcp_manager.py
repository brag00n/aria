import asyncio
import os
import json
import components.utils as utils
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

class McpManager:
    """Gère la connexion aux serveurs MCP et la conversion des outils pour le LLM."""
    
    def __init__(self, mcp_params):
        self.params = mcp_params or {}
        self.servers_conf = self.params.get("servers", {})
        self.tool_registry = {}  # Mappe tool_name -> session
        self.openai_tools = []   # Liste des schémas d'outils format OpenAI
        self.stack = AsyncExitStack()
        self.loop = asyncio.new_event_loop()
        
    def start(self):
        """Initialise les connexions aux serveurs de manière synchrone pour le démarrage d'Aria."""
        utils.log_perf("MCP", "Initialisation des serveurs MCP...")
        try:
            self.loop.run_until_complete(self._connect_all())
        except Exception as e:
            utils.log_perf("MCP", f"Erreur critique démarrage: {e}")

    async def _connect_all(self):
        """Parcourt la config et connecte chaque serveur (Stdio ou SSE)."""
        for name, conf in self.servers_conf.items():
            try:
                session = None
                # Cas 1 : Serveur Local (stdio via commande)
                if "run" in conf or "command" in conf:
                    cmd = conf.get("run") or conf.get("command")
                    args = conf.get("args", [])
                    env = os.environ.copy()
                    if "env" in conf:
                        env.update(conf["env"])
                    
                    params = StdioServerParameters(command=cmd, args=args, env=env)
                    read, write = await self.stack.enter_async_context(stdio_client(params))
                    session = await self.stack.enter_async_context(ClientSession(read, write))

                # Cas 2 : Serveur Distant (SSE)
                elif "url" in conf:
                    headers = conf.get("headers", {})
                    read, write = await self.stack.enter_async_context(sse_client(url=conf["url"], headers=headers))
                    session = await self.stack.enter_async_context(ClientSession(read, write))

                if session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    
                    # Enregistrement et conversion
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
                    utils.log_perf("MCP", f"Serveur [{name}] chargé : {len(tools_resp.tools)} outils.")
            
            except Exception as e:
                utils.log_perf("MCP", f"Impossible de connecter [{name}]: {e}")

    def get_tools_schema(self):
        """Retourne la liste des outils pour llama-cpp."""
        return self.openai_tools if self.openai_tools else None

    def call_tool_sync(self, tool_name, arguments):
        """Exécute un outil de manière synchrone."""
        if tool_name not in self.tool_registry:
            return f"Erreur : L'outil {tool_name} n'existe pas."
        
        session = self.tool_registry[tool_name]
        try:
            utils.log_perf("MCP", f"Exécution de {tool_name} avec {arguments}...")
            result = self.loop.run_until_complete(
                session.call_tool(tool_name, arguments=arguments)
            )
            # On suppose que le contenu est textuel
            return result.content[0].text
        except Exception as e:
            return f"Erreur lors de l'appel de l'outil: {str(e)}"

    def shutdown(self):
        """Ferme proprement toutes les connexions."""
        self.loop.run_until_complete(self.stack.aclose())
        self.loop.close()