import os
import components.utils as utils
from .utils import TtsQueue, StepTimer
from .mcp import Mcp

MAX_TOOL_ROUNDS = 5

class LlmConnectAnthropic:
    def __init__(self, params, all_config=None):
        self.params = params or {}
        self.backend_type = "anthropic"
        self.model_name = self.params.get("model_name", "claude-haiku-5-5")
        self.base_url = self.params.get("base_url") or None
        self.apikey = utils.expand_secrets(self.params.get("apikey") or "") or os.environ.get("ANTHROPIC_API_KEY")
        self.system_message = self.params.get("system_message", "")
        self.max_tokens = self.params.get("max_tokens", 2048)
        self.verbose = self.params.get("verbose", False)

        utils.log_info("LLM-anthropic", f"Initialisation du mode anthropic 🌐 (modèle {self.model_name}, url {self.base_url or 'défaut'})")
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            utils.log_info("LLM-anthropic", "❌ Erreur: anthropic manquant. pip install anthropic")
            raise

        if not self.apikey:
            raise ValueError("Clé API Anthropic manquante (params.apikey ou variable ANTHROPIC_API_KEY)")

        self.client = AsyncAnthropic(api_key=self.apikey, base_url=self.base_url, timeout=60.0)
        self.user_aware_messages = {}

        self.mcp_manager = None
        if all_config and "mcp" in all_config:
            self.mcp_manager = Mcp(all_config["mcp"]["params"])

    async def _get_tools(self):
        """Convertit les outils MCP (format OpenAI) au format Anthropic."""
        openai_tools = await self.mcp_manager.get_tools_for_openai() if self.mcp_manager else None
        if not openai_tools:
            return None
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description") or "",
                "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
            }
            for t in openai_tools
        ]

    async def get_answer_web(self, tts, query, user):
        history = self.user_aware_messages.setdefault(user, [])
        start_len = len(history)
        history.append({"role": "user", "content": query})
        timer = StepTimer("LLM-anthropic")

        try:
            tools = await self._get_tools()
            timer.mark("outils_mcp", f"{len(tools) if tools else 0} outil(s)")
            full_res = ""
            first_token = False
            queue = TtsQueue(tts, user, "LLM-anthropic")

            for round_idx in range(MAX_TOOL_ROUNDS):
                kwargs = {
                    "model": self.model_name,
                    "max_tokens": self.max_tokens,
                    "messages": history,
                }
                if self.system_message:
                    kwargs["system"] = self.system_message
                if tools:
                    kwargs["tools"] = tools

                tts_buf = ""
                async with self.client.messages.stream(**kwargs) as stream:
                    timer.mark(f"requete_llm_{round_idx + 1}")
                    async for content in stream.text_stream:
                        if not first_token:
                            first_token = True
                            timer.mark("premier_token")
                        full_res += content
                        tts_buf += content

                        tts_buf = queue.feed(tts_buf)  # synthèse en arrière-plan, le flux LLM continue

                        ready = queue.pop_ready()
                        for audio, spoken in ready:
                            yield full_res, audio, spoken
                        if not ready:
                            yield full_res, None, None
                    final = await stream.get_final_message()

                # Reste du buffer de ce tour
                queue.flush(tts_buf)

                # Les blocs (thinking, tool_use...) sont renvoyés tels quels
                history.append({"role": "assistant", "content": final.content})

                if final.stop_reason != "tool_use":
                    if final.stop_reason == "refusal":
                        utils.log_info("LLM-anthropic", "⚠️ Réponse refusée par le modèle")
                    break

                results = []
                for block in final.content:
                    if block.type != "tool_use":
                        continue
                    utils.log_info("LLM-anthropic", f"🛠️ Action MCP : {block.name}({block.input})")
                    full_res += utils.tool_notice(block.name)      # affiché dans le chat, jamais lu à voix haute
                    yield full_res, None, None
                    result = await self.mcp_manager.call_tool(block.name, block.input)
                    timer.mark(f"outil_{block.name}")
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                history.append({"role": "user", "content": results})

            timer.mark("fin_texte_llm")
            # Blocs encore en cours de synthèse
            async for audio, spoken in queue.drain():
                yield full_res, audio, spoken
            timer.mark("fin_generation", f"TTS cumulé {queue.total:.2f}s sur {queue.count} bloc(s)")

        except GeneratorExit:
            # Réponse interrompue (bouton / mot-clé « stop ») : on retire ce tour, sans appel d'outil resté sans résultat
            del history[start_len:]
            raise
        except Exception as e:
            utils.log_info("LLM-anthropic", f"❌ Erreur : {type(e).__name__}: {str(e)[:300]}")
            # On retire les messages de ce tour pour ne pas laisser un historique incohérent
            del history[start_len:]
            yield f"Erreur LLM: {str(e)}", None, None
