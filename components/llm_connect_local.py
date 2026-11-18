# components/llm_connect_local.py
"""Moteur de langage LOCAL (llama-cpp). Mêmes capacités que les moteurs distants (llm_connect_openai.py), dont il reprend la forme :
  - historique PAR utilisateur (`user_aware_messages`, borné par components/history.py), donc une conversation par session d'agent ;
  - consigne propre à la requête (`contexte`, par exemple l'interlocuteur actuel) ajoutée au message système, jamais cumulée ;
  - outils MCP en asynchrone (`mcp_manager`), comme les autres moteurs : plus de `Mcp.start()` ni d'appel synchrone ;
  - synthèse vocale au fil de l'eau (`TtsQueue`), ligne « outil utilisé » dans le texte affiché, arrêt propre (GeneratorExit) et message d'erreur « Erreur LLM: ».
llama-cpp est SYNCHRONE : chaque appel se fait dans un thread (`run_in_executor`) et sous un verrou (un seul appel du modèle à la fois), pour ne jamais bloquer
la boucle asynchrone, qui sert aussi les autres sessions et l'API.
"""
import asyncio
import json
import os
import threading
import uuid

import components.utils as utils
from components import history as history_mod
from .utils import StepTimer, TtsQueue

MAX_TOOL_ROUNDS = 5
_FIN = object()


class LlmConnectLocal:
    def __init__(self, params, all_config=None, llama_factory=None):
        self.params = params or {}
        self.system_message = self.params.get("system_message", "")
        self.chat_format = self.params.get("chat_format", "mistral-instruct")
        self.context_length = self.params.get("context_length", 4096)
        self.verbose = self.params.get("verbose", False)
        self.memory = (all_config or {}).get("memory")          # plafonds de l'historique (components/history.py)
        self.backend_type = "local"
        self.user_aware_messages = {}
        self._llama_lock = threading.Lock()                     # un seul appel du modèle à la fois (llama-cpp n'est pas réentrant)

        # Réglages d'échantillonnage : ils se passent à la génération, pas au constructeur de Llama
        # (le constructeur les ignorerait sans erreur). Une valeur absente de la config n'est pas envoyée.
        self.gen_kwargs = {k: self.params[k] for k in ("temperature", "repeat_penalty", "stop") if k in self.params}

        if llama_factory is not None:                           # tests : un faux modèle remplace llama-cpp et le fichier .gguf
            self.llm = llama_factory(self.params)
        else:
            self.llm = self._charger_modele()

        # Outils : même gestionnaire asynchrone que les moteurs distants
        from .mcp import Mcp
        self.mcp_manager = None
        if all_config and "mcp" in all_config:
            self.mcp_manager = Mcp(all_config["mcp"]["params"], secret_for_servers=all_config.get("_memory_secret", ""),
                                   hosted_only=((all_config.get("agents") or all_config.get("access") or {}).get("hosted_only") or ()))
        self.mcp = self.mcp_manager                               # ancien nom, conservé

    def _charger_modele(self):
        from llama_cpp import Llama
        model_file = self.params.get("model_file", "ministral-3b-instruct.Q4_K_M.gguf")
        models_dir = self.params.get("custom_path") or utils.aria_path("models", "llm")
        model_path = os.path.join(models_dir, model_file)
        n_gpu_layers = self.params.get("num_gpu_layers", -1)

        # Fichier absent : téléchargement depuis Hugging Face si "hf_repo" est renseigné (ex. "owner/nom-GGUF")
        hf_repo = self.params.get("hf_repo")
        if hf_repo and not os.path.isfile(model_path):
            from huggingface_hub import hf_hub_download
            os.makedirs(models_dir, exist_ok=True)
            utils.log_info("LLM-LOCAL", f"Modèle absent, téléchargement de {hf_repo}/{model_file} (plusieurs Go, patientez)...")
            hf_hub_download(repo_id=hf_repo, filename=model_file, local_dir=models_dir)
            utils.log_info("LLM-LOCAL", "Téléchargement terminé")

        utils.log_info("LLM-LOCAL", f"Chargement du modèle : {model_path}")
        return Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            chat_format=self.chat_format,
            n_ctx=self.context_length,
            n_batch=self.params.get("n_batch", 512),            # vitesse de traitement du prompt
            flash_attn=self.params.get("flash_attn", False),    # GPU de capacité 8.0+ seulement (Ampere, Ada...)
            verbose=self.verbose
        )

    # ------------------------------------------------------------------ appels du modèle (synchrones, dans un thread)
    def _completion(self, messages, tools=None, stream=False):
        """Un appel du modèle, SOUS verrou. Sans flux : la réponse complète. Avec flux : un générateur de morceaux, lui aussi consommé sous verrou."""
        kwargs = dict(messages=messages, **self.gen_kwargs)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if stream:
            kwargs["stream"] = True
        with self._llama_lock:
            return self.llm.create_chat_completion(**kwargs)

    async def _flux(self, messages):
        """Morceaux d'une réponse en flux. Chaque `next()` du générateur llama s'exécute dans un thread, sous le verrou : la boucle reste libre entre deux jetons."""
        boucle = asyncio.get_running_loop()
        flux = await boucle.run_in_executor(None, lambda: self._completion(messages, stream=True))
        iterateur = iter(flux)

        def suivant():
            with self._llama_lock:
                return next(iterateur, _FIN)
        try:
            while True:
                morceau = await boucle.run_in_executor(None, suivant)
                if morceau is _FIN:
                    return
                yield morceau
        finally:
            close = getattr(flux, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ un tour
    async def get_answer_web(self, tts, query, user, contexte=""):
        tag = "LLM-LOCAL"
        timer = StepTimer(tag)
        queue = TtsQueue(tts, user, tag)

        messages = self.user_aware_messages.setdefault(user, [{"role": "system", "content": self.system_message}])
        # `contexte` : consigne propre à cette requête (ex. interlocuteur actuel). Le message système est réécrit à chaque requête (jamais cumulé).
        messages[0]["content"] = f"{self.system_message}\n\n{contexte}".strip() if contexte else self.system_message
        history_mod.shorten_tool_results(messages, self.memory, first=1)
        history_mod.trim(messages, self.memory, first=1)
        start_len = len(messages)
        messages.append({"role": "user", "content": query})
        boucle = asyncio.get_running_loop()

        try:
            tools = await self.mcp_manager.get_tools_for_openai() if self.mcp_manager else None
            timer.mark("outils_mcp", f"{len(tools) if tools else 0} outil(s)")
            full_res = ""

            for round_idx in range(MAX_TOOL_ROUNDS):
                dernier = round_idx == MAX_TOOL_ROUNDS - 1
                round_text, tts_buf, appels = "", "", []

                if tools and not dernier:
                    # Étape 1 : un appel SANS flux pour détecter une demande d'outil (llama-cpp ne reconstitue pas les outils d'un flux)
                    reponse = await boucle.run_in_executor(None, lambda: self._completion(messages, tools=tools))
                    choix = reponse["choices"][0]["message"]
                    appels = choix.get("tool_calls") or []
                    if not appels:
                        round_text = choix.get("content") or ""
                        full_res += round_text
                        tts_buf = queue.feed(round_text)
                        queue.flush(tts_buf)
                        for audio, spoken in queue.pop_ready():
                            yield full_res, audio, spoken
                        yield full_res, None, None
                    timer.mark(f"requete_llm_{round_idx + 1}")
                else:
                    # Dernier tour, ou aucun outil : réponse en flux
                    async for morceau in self._flux(messages):
                        delta = morceau["choices"][0].get("delta") or {}
                        content = delta.get("content")
                        if not content:
                            continue
                        full_res += content
                        round_text += content
                        tts_buf += content
                        tts_buf = queue.feed(tts_buf)
                        ready = queue.pop_ready()
                        for audio, spoken in ready:
                            yield full_res, audio, spoken
                        if not ready:
                            yield full_res, None, None
                    queue.flush(tts_buf)
                    timer.mark(f"requete_llm_{round_idx + 1}")

                if not appels:
                    if round_text.strip():
                        messages.append({"role": "assistant", "content": round_text})
                    elif messages[-1]["role"] == "user":
                        messages.pop()                                    # tour sans réponse : on retire aussi la question
                    break

                # Exécution des outils demandés, puis nouveau tour
                calls = []
                for c in appels:
                    f = c.get("function") or {}
                    ident = c.get("id") or ""
                    calls.append({"id": ident if len(ident) >= 9 else f"call{uuid.uuid4().hex[:9]}", "type": "function",
                                  "function": {"name": f.get("name", ""), "arguments": f.get("arguments") or "{}"}})
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    utils.log_info(tag, f"🛠️ Action MCP : {name}({args})")
                    full_res += utils.tool_notice(name)                  # affiché dans le chat, jamais lu à voix haute
                    yield full_res, None, None
                    result = await self.mcp_manager.call_tool(name, args) if self.mcp_manager else f"Erreur : aucun outil disponible pour {name}."
                    timer.mark(f"outil_{name}")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})

            async for audio, spoken in queue.drain():
                yield full_res, audio, spoken
            timer.mark("fin_generation", f"TTS cumulé {queue.total:.2f}s sur {queue.count} bloc(s)")

        except GeneratorExit:
            del messages[start_len:]                                      # réponse interrompue : on retire ce tour
            raise
        except Exception as e:
            utils.log_info(tag, f"❌ Erreur : {str(e)}")
            del messages[start_len:]
            yield f"Erreur LLM: {str(e)}", None, None
