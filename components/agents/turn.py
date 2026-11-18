"""Cœur d'un tour de conversation d'un agent : indépendant du transport.

`run_turn` est un générateur asynchrone d'ÉVÉNEMENTS `(nom, données)` ; il ne sait rien de SSE, de HTTP ni de Gradio. Deux formateurs le consomment :
  - le flux SSE de l'API (`api._stream_turn`) : chaque événement devient `event: nom / data: JSON`, le fichier wav d'un événement `audio` est lu et encodé en base64 ;
  - le chat de l'interface d'Aria (`app.py`), qui garde son propre pont audio : il reçoit le CHEMIN du wav et le lit lui-même.
Un seul chemin pour les deux : une règle (mots-clés, audit, arrêt, erreurs) ne se corrige qu'ici.

Événements : state{state}, warning{code,message}, user_text{text}, conversation{state}, tool{name}, text{text}, audio{path,text}, done{...}, error{code,message}.
Un tour se termine TOUJOURS par `done` ou `error` (jamais les deux, jamais rien).
"""
import asyncio
import base64
import re
import time

import components.utils as utils
from components import echanges

TOOL_NOTICE_RE = re.compile(r"\n\n> 🛠️ Outil utilisé : `([^`]+)`\n\n")


class AudioTooLong(Exception):
    pass


def decode_and_transcribe(deps, raw):
    """Décodage puis STT, dans un thread : ni l'un ni l'autre ne doit bloquer la boucle asynchrone."""
    samples = deps.decode_audio(raw)
    if len(samples) / 16000.0 > deps.max_audio_s:
        raise AudioTooLong()
    if deps.stt_lock:
        with deps.stt_lock:
            return deps.stt.transcribe_translate(samples)
    return deps.stt.transcribe_translate(samples)


async def run_turn(deps, session, rec, text, audio, speak, audit, ignored=False, contexte="", qui=None):
    """Un tour. `text` : texte saisi (ou None) ; `audio` : data URI base64 (ou None) ; `speak` : synthèse vocale demandée ; `audit` : fonction d'audit de l'API ;
    `ignored` : les consignes de l'application envoyées sont ignorées (la session existe déjà) ; `contexte` : consigne propre à ce tour (par exemple l'interlocuteur
    d'Aria) ; `qui` : interlocuteur écrit dans le journal d'audit (par défaut celui de la session, sinon le nom de l'agent).
    Un seul tour à la fois par session : si un tour est en cours, `error{busy}` immédiat."""
    if session.lock.locked():
        yield "error", {"code": "busy", "message": "un tour est déjà en cours dans cette session"}
        return
    async with session.lock:
        session.stop.clear()
        session.touch()
        t0 = time.perf_counter()
        try:
            yield "state", {"state": "thinking"}
            if ignored:
                yield "warning", {"code": "system_ignored", "message": "la session existe déjà : ses consignes sont fixées à sa création "
                                                                       "(utilisez une autre session pour changer de consignes)"}
            if audio is not None:
                try:
                    raw = base64.b64decode(audio.split(",", 1)[1], validate=False)
                    text = await asyncio.get_running_loop().run_in_executor(None, decode_and_transcribe, deps, raw)
                except AudioTooLong:
                    yield "error", {"code": "audio_too_long", "message": f"audio limité à {deps.max_audio_s:.0f} s"}
                    return
                except Exception as e:
                    yield "error", {"code": "bad_audio", "message": "audio illisible"}
                    utils.log_info("AGENTS", f"audio illisible ({rec['id']}) : {type(e).__name__}")
                    return
                if not re.search(r"\w", text or ""):
                    yield "done", {"empty": True}
                    return
                yield "user_text", {"text": text}
            action = "say"
            if session.conv is not None:
                # Option « conversation » du profil : mots d'arrêt, d'ouverture et de clôture (components/conversation.py). Absente = jamais ce bloc, flux inchangé.
                action = session.conv.classify(text, from_voice=audio is not None)
                if action == "stop":
                    session.stop.set()
                    yield "done", {"stopped": True, "reason": "keyword"}
                    return
                if action == "ignore":
                    yield "done", {"ignored": True}                       # hors conversation : ni LLM, ni historique, ni audit d'échange
                    return
                if action == "open":
                    session.conv.begin()
                    yield "conversation", {"state": "open"}
            audit("turn", key=rec["id"], session=session.sid, voice=audio is not None)
            tts = deps.tts if speak else None
            vu = ""                                   # texte déjà émis, pour n'envoyer que les changements
            outils = set()
            # Journal d'audit : source = identifiant interne de l'agent + son nom du moment ; l'échange est propre à CETTE tâche (tours concurrents sans mélange)
            # `who` : la personne qui parle (agent qui publie les informations de son interlocuteur), sinon le nom de l'agent ; `src` garde TOUJOURS l'agent
            if qui is None:
                qui = ((session.speaker or {}).get("nom") or "Inconnue") if rec.get("publish_speaker") else (rec.get("label") or rec["id"])
            echanges.start(echanges.source_label(rec["id"], rec.get("label")), qui, session.conv_id, voix=audio is not None, query=text)
            gen = session.engine.get_answer_web(tts, text, session.user, contexte) if contexte else session.engine.get_answer_web(tts, text, session.user)
            try:
                async for full, wav, spoken in gen:
                    if session.stop.is_set():
                        yield "done", {"stopped": True}
                        return
                    propre = TOOL_NOTICE_RE.sub("", full or "")
                    for name in TOOL_NOTICE_RE.findall(full or ""):
                        if name not in outils:
                            outils.add(name)
                            yield "tool", {"name": name}
                    if full and full.startswith("Erreur LLM:"):
                        yield "error", {"code": "llm", "message": "le modèle n'a pas pu répondre"}
                        utils.log_info("AGENTS", f"{rec['id']} : {full[:200]}")
                        return
                    if propre != vu:
                        vu = propre
                        yield "text", {"text": propre}
                    if wav:
                        yield "state", {"state": "speaking"}
                        yield "audio", {"path": wav, "text": spoken or ""}
            finally:
                await gen.aclose()
            if action == "close" and session.conv is not None:
                session.conv.end()
                deps.sessions.reset(session)                                # clôture : la conversation est oubliée (nouvelle époque pour le journal d'audit)
                yield "conversation", {"state": "closed"}
            yield "done", {"ms": int((time.perf_counter() - t0) * 1000)}
        except asyncio.CancelledError:                # client parti : on arrête proprement
            raise
        except Exception as e:
            utils.log_info("AGENTS", f"❌ tour ({rec['id']}) : {type(e).__name__}: {str(e)[:200]}")
            yield "error", {"code": "internal", "message": "erreur interne"}
