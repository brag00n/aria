"""Pont entre deux boucles asyncio, pour l'agent par défaut `aria`.

L'agent `aria` n'a pas son propre moteur : il réutilise le `Llm` et le gestionnaire MCP GLOBAUX d'Aria, qui sont connectés dans la boucle de l'interface (Gradio).
Or l'API des agents (port 7861) tourne dans une AUTRE boucle (un thread uvicorn). Le client MCP (anyio) exige d'ouvrir et de fermer ses connexions dans la même tâche : un
tour de l'agent `aria` doit donc TOUJOURS s'exécuter dans la boucle de l'interface. `run_in_loop` y exécute un générateur d'événements et les rend, un à un, à l'appelant
(qui peut être dans une autre boucle) par une file thread-safe ; l'arrêt de l'appelant (client parti) ferme proprement le générateur d'origine.
"""
import asyncio
import threading

_FIN = object()


class LoopHolder:
    """Boucle de l'interface, posée par `app.py` (au démarrage de Gradio). `None` tant qu'elle n'est pas connue."""

    def __init__(self):
        self.loop = None
        self._lock = threading.Lock()

    def set(self, loop):
        with self._lock:
            self.loop = loop

    def get(self):
        with self._lock:
            return self.loop


async def run_in_loop(holder, factory):
    """Itère `factory()` (un générateur asynchrone) DANS la boucle de `holder` et rend ses éléments dans la boucle courante.
    Si la boucle de l'interface n'est pas encore connue, ou est la boucle courante, le générateur est simplement itéré ici (pas de pont)."""
    cible = holder.get()
    courante = asyncio.get_running_loop()
    if cible is None or cible is courante or not cible.is_running():
        async for element in factory():
            yield element
        return

    file = asyncio.Queue()
    arret = threading.Event()

    async def pomper():
        gen = factory()
        try:
            async for element in gen:
                if arret.is_set():
                    break
                courante.call_soon_threadsafe(file.put_nowait, ("element", element))
        except BaseException as e:                                     # l'erreur est rendue à l'appelant, dans SA boucle
            courante.call_soon_threadsafe(file.put_nowait, ("erreur", e))
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
            courante.call_soon_threadsafe(file.put_nowait, ("fin", _FIN))

    futur = asyncio.run_coroutine_threadsafe(pomper(), cible)
    try:
        while True:
            genre, valeur = await file.get()
            if genre == "fin":
                return
            if genre == "erreur":
                raise valeur
            yield valeur
    finally:
        arret.set()                                                    # l'appelant est parti : le générateur d'origine s'arrête au prochain élément
        if not futur.done():
            futur.cancel()
