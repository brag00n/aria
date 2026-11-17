import json
import uuid
import components.utils as utils
from .utils import StepTimer, TtsQueue
from .mcp import Mcp
from . import history as history_mod

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
        self.memory = (all_config or {}).get("memory")        # plafonds de l'historique (components/history.py)

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
            self.mcp_manager = Mcp(all_config["mcp"]["params"], secret_for_servers=all_config.get("_memory_secret", ""),
                                  hosted_only=((all_config.get("agents") or all_config.get("access") or {}).get("hosted_only") or ()))


    async def get_answer_web(self, tts, query, user, contexte=""):
        tag = f"LLM-{self.backend_type}"
        timer = StepTimer(tag)
        queue = TtsQueue(tts, user, tag)

        messages = self.user_aware_messages.setdefault(
            user, [{"role": "system", "content": self.system_message}]
        )
        # `contexte` : consigne propre à cette requête (ex. interlocuteur actuel). Le message système est réécrit à chaque
        # requête (jamais cumulé) ; les messages suivants de l'historique ne sont pas touchés.
        messages[0]["content"] = f"{self.system_message}\n\n{contexte}".strip() if contexte else self.system_message
        # Historique borné AVANT d'ajouter la question (start_len, utilisé pour annuler un tour en erreur, reste exact)
        history_mod.shorten_tool_results(messages, self.memory, first=1)
        history_mod.trim(messages, self.memory, first=1)
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
                # Dernier tour autorisé : plus d'outils, pour obliger le modèle à répondre avec ce qu'il a déjà obtenu
                # (sinon la boucle s'arrête sur un appel d'outil et l'utilisateur n'a aucune réponse)
                last_round = round_idx == MAX_TOOL_ROUNDS - 1
                if tools and not last_round:
                    kwargs["tools"] = tools
                elif last_round and round_idx > 0:
                    utils.log_info(tag, f"Tour {round_idx + 1} : limite d'outils atteinte, réponse demandée sans outil")

                stream = await self.client.chat.completions.create(**kwargs)
                timer.mark(f"requete_llm_{round_idx + 1}")

                round_text = ""
                tts_buf = ""
                tool_calls = {}
                finish_reason = None

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
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

                # Diagnostic : fin de tour et appels d'outils reçus (un modèle qui annonce « je cherche » sans appeler l'outil se voit ici)
                utils.log_info(tag, f"Tour {round_idx + 1} : finish_reason={finish_reason}, "
                                    f"{len(tool_calls)} appel(s) d'outil{' : ' + ', '.join(c['name'] for c in tool_calls.values()) if tool_calls else ''}")

                if not tool_calls:
                    # Pas de message assistant vide : certains modèles de conversation (Mistral, Ministral) refusent
                    # un message sans texte ni appel d'outil, et l'historique conservé bloquerait toutes les requêtes suivantes
                    if round_text.strip():
                        messages.append({"role": "assistant", "content": round_text})
                    elif messages[-1]["role"] == "user":
                        messages.pop()             # tour sans réponse : on retire aussi la question, l'alternance reste correcte
                    break

                # Exécution des outils demandés, puis nouveau tour
                calls = []
                for idx in sorted(tool_calls):
                    c = tool_calls[idx]
                    calls.append({
                        # Identifiant d'au moins 9 caractères : exigé par le modèle de conversation de Mistral Small 3.2
                        "id": c["id"] if len(c["id"]) >= 9 else f"call{uuid.uuid4().hex[:9]}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    })
                messages.append({"role": "assistant", "content": round_text or "", "tool_calls": calls})

                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    utils.log_info(tag, f"🛠️ Action MCP : {name}({args})")
                    full_res += utils.tool_notice(name)            # affiché dans le chat, jamais lu à voix haute
                    yield full_res, None, None
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

        except GeneratorExit:
            # Réponse interrompue (bouton / mot-clé « stop ») : on retire ce tour, sans appel d'outil resté sans résultat
            del messages[start_len:]
            raise
        except Exception as e:
            utils.log_info(tag, f"❌ Erreur : {str(e)}")
            # On retire les messages de ce tour pour ne pas laisser un historique incohérent
            del messages[start_len:]
            # On renvoie 3 valeurs au lieu de 2 pour correspondre à ce que app.py attend
            yield f"Erreur LLM: {str(e)}", None, None
