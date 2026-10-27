import json
import components.utils as utils
from .utils import StepTimer, TtsQueue
from .mcp import Mcp

MAX_TOOL_ROUNDS = 5

class LlmConnectOpenai:
    def __init__(self, params, all_config=None):
        self.params = params or {}
        self.backend_type=self.params.get("backend_type", "local")
        self.base_url = self.params.get("base_url", "http://localhost:1234/v1")
        self.model_name = self.params.get("model_name", "local-model")
        self.apikey = self.params.get("apikey", f"{self.backend_type}")
        self.system_message = self.params.get("system_message", "")
        self.verbose = self.params.get("verbose", False)

        utils.log_info(f"LLM-{self.backend_type}", f"Initialisation du mode {self.backend_type} 🌐")
        utils.log_info(f"LLM-{self.backend_type}", f"Utilisation du modele {self.model_name} via l'url {self.base_url}")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            utils.log_info(f"LLM-{self.backend_type}", "❌ Erreur: openai manquant. pip install openai")
            raise

        # Préparation de l'acces au modele
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=f"{self.apikey}",timeout=60.0)
        self.user_aware_messages = {}

        # Préparation des outils
        self.mcp_manager = None
        if all_config and "mcp" in all_config:
            self.mcp_manager = Mcp(all_config["mcp"]["params"])


    async def get_answer_web(self, tts, query, user):
        tag = f"LLM-{self.backend_type}"
        timer = StepTimer(tag)
        queue = TtsQueue(tts, user, tag)

        messages = self.user_aware_messages.setdefault(
            user, [{"role": "system", "content": self.system_message}]
        )
        start_len = len(messages)
        messages.append({"role": "user", "content": query})

        try:
            # 1. Préparation des outils
            tools = await self.mcp_manager.get_tools_for_openai() if self.mcp_manager else None
            timer.mark("outils_mcp", f"{len(tools) if tools else 0} outil(s)")

            full_res = ""
            first_token = False

            # Une seule requête en streaming par tour : le texte est parlé au fil de l'eau,
            # et un éventuel appel d'outil est reconstitué depuis les deltas.
            for round_idx in range(MAX_TOOL_ROUNDS):
                kwargs = {"model": self.model_name, "messages": messages, "stream": True}
                if tools:
                    kwargs["tools"] = tools

                stream = await self.client.chat.completions.create(**kwargs)
                timer.mark(f"requete_llm_{round_idx + 1}")

                round_text = ""
                tts_buf = ""
                tool_calls = {}

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    for tc in (delta.tool_calls or []):
                        slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] = tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

                    content = delta.content
                    if not content:
                        continue
                    if not first_token:
                        first_token = True
                        timer.mark("premier_token")

                    full_res += content
                    round_text += content
                    tts_buf += content

                    tts_buf = queue.feed(tts_buf)  # synthèse en arrière-plan, le flux LLM continue

                    ready = queue.pop_ready()
                    for audio, spoken in ready:
                        yield full_res, audio, spoken
                    if not ready:
                        yield full_res, None, None

                # Reste du buffer de ce tour
                queue.flush(tts_buf)

                if not tool_calls:
                    messages.append({"role": "assistant", "content": round_text})
                    break

                # Exécution des outils demandés, puis nouveau tour
                calls = []
                for idx in sorted(tool_calls):
                    c = tool_calls[idx]
                    calls.append({
                        "id": c["id"] or f"call_{round_idx}_{idx}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    })
                messages.append({"role": "assistant", "content": round_text or None, "tool_calls": calls})

                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    utils.log_info(tag, f"🛠️ Action MCP : {name}({args})")
                    if self.mcp_manager:
                        result = await self.mcp_manager.call_tool(name, args)
                    else:
                        result = f"Erreur : aucun outil disponible pour {name}."
                    timer.mark(f"outil_{name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result
                    })

            timer.mark("fin_texte_llm")
            # Blocs encore en cours de synthèse
            async for audio, spoken in queue.drain():
                yield full_res, audio, spoken
            timer.mark("fin_generation", f"TTS cumulé {queue.total:.2f}s sur {queue.count} bloc(s)")

        except Exception as e:
            utils.log_info(tag, f"❌ Erreur : {str(e)}")
            # On retire les messages de ce tour pour ne pas laisser un historique incohérent
            del messages[start_len:]
            # On renvoie 3 valeurs au lieu de 2 pour correspondre à ce que app.py attend
            yield f"Erreur LLM: {str(e)}", None, None
