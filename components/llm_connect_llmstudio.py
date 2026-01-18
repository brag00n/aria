# components/llm_connect_llmstudio.py
import asyncio
import json
import components.utils as utils
from .utils import clean_text_for_tts, remove_nonverbal_cues
from .mcp_manager import McpManager

class LlmConnectLlmStudio:
    def __init__(self, params, all_config=None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            utils.log_perf("LLM-STUDIO", "❌ Erreur: openai manquant. pip install openai")
            raise

        self.params = params or {}
        self.base_url = self.params.get("base_url", "http://localhost:1234/v1")
        self.model_name = self.params.get("model_name", "local-model")
        self.system_message = self.params.get("system_message", "")
        
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="lm-studio")
        self.user_aware_messages = {}

        self.mcp_manager = None
        if all_config and "Mcp" in all_config:
            self.mcp_manager = McpManager(all_config["Mcp"]["params"])

    async def get_answer_web(self, tts, query, user):
        if user not in self.user_aware_messages:
            self.user_aware_messages[user] = [{"role": "system", "content": self.system_message}]

        self.user_aware_messages[user].append({"role": "user", "content": query})

        # 1. Préparation des outils
        tools = await self.mcp_manager.get_tools_for_openai() if self.mcp_manager else None

        try:
            # 2. Premier passage : Détection d'outil
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.user_aware_messages[user],
                tools=tools,
                tool_choice="auto" if tools else None
            )

            msg = response.choices[0].message
            
            # 3. Exécution de l'outil si nécessaire
            if msg.tool_calls:
                self.user_aware_messages[user].append(msg)
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    utils.log_perf("LLM-STUDIO", f"🛠️ Action MCP : {name}({args})")
                    result = await self.mcp_manager.call_tool(name, args)
                    
                    self.user_aware_messages[user].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result
                    })

            # 4. Passage final : Streaming de la réponse et du TTS
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.user_aware_messages[user],
                stream=True
            )

            full_res = ""
            tts_buf = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_res += content
                    tts_buf += content
                    
                    if any(p in content for p in [".", "!", "?", "\n"]):
                        txt = clean_text_for_tts(remove_nonverbal_cues(tts_buf)).strip()
                        if len(txt) > 2:
                            audio = tts.run_tts_to_file(txt, user_id=user)
                            yield full_res, audio
                            tts_buf = ""
                        else:
                            yield full_res, None
                    else:
                        yield full_res, None

            # Reste du buffer final
            txt_fin = clean_text_for_tts(remove_nonverbal_cues(tts_buf)).strip()
            if len(txt_fin) > 1:
                yield full_res, tts.run_tts_to_file(txt_fin, user_id=user)

            self.user_aware_messages[user].append({"role": "assistant", "content": full_res})

        except Exception as e:
            utils.log_perf("LLM-STUDIO", f"❌ Erreur : {e}")
            yield "Oups, j'ai eu un souci avec mes outils.", None