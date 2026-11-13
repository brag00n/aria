/*
 * Client JavaScript d'Aria pour les pages web tierces. Un seul fichier, sans dépendance à l'interface d'Aria.
 *
 *   <script src="https://ARIA:7861/aria-client.js"></script>
 *   <script>
 *     const aria = AriaClient.create({ baseUrl: "https://ARIA:7861", apiKey: "ak_…" });
 *     aria.on("text", t => zone.textContent = t);        // texte de la réponse (cumulé)
 *     aria.on("state", s => …);                          // idle | listening | hearing | thinking | speaking
 *     await aria.start();                                // écoute le micro (nécessite une page HTTPS ou localhost)
 *     aria.say("Quelle heure est-il ?");                 // saisie texte
 *     aria.interrupt();  aria.stop();  aria.destroy();
 *   </script>
 *
 * Événements : state, text, user_text, tool, audio-start, audio-end, done, error({code, message}).
 * Options : baseUrl, apiKey, voice (true), speak (true : lire la réponse), session ("default"), vad {positive, negative, redemption, minFrames},
 *           modelUrl (modèle Silero, par défaut servi par Aria), cdn {ort, vad} (adresses des bibliothèques du VAD),
 *           debug (false : trace dans la console les événements du VAD et les envois, utile quand l'écoute ne réagit pas).
 * Le VAD (détection de parole) est exécuté dans le navigateur : seule la phrase détectée est envoyée à Aria.
 */
(function (root) {
  'use strict';

  var DEFAULT_CDN = {
    ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js',
    vad: 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js'
  };

  // Découpe un flux Server-Sent Events : retourne les événements complets et le reste du tampon.
  function parseSse(buffer) {
    var events = [];
    var rest = String(buffer).replace(/\r\n/g, '\n');
    var idx;
    while ((idx = rest.indexOf('\n\n')) >= 0) {
      var block = rest.slice(0, idx);
      rest = rest.slice(idx + 2);
      var name = 'message';
      var data = [];
      block.split('\n').forEach(function (line) {
        if (line.indexOf('event:') === 0) name = line.slice(6).trim();
        else if (line.indexOf('data:') === 0) data.push(line.slice(5).replace(/^ /, ''));
      });
      if (data.length) {
        try { events.push({ event: name, data: JSON.parse(data.join('\n')) }); } catch (e) { /* bloc illisible : ignoré */ }
      }
    }
    return { events: events, rest: rest };
  }

  function loadScriptDefault(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('chargement impossible : ' + src)); };
      document.head.appendChild(s);
    });
  }

  function create(options) {
    var o = Object.assign({ voice: true, speak: true, session: 'default', vad: {}, cdn: {} }, options || {});
    if (!o.baseUrl || !o.apiKey) throw new Error('AriaClient : baseUrl et apiKey sont obligatoires');
    var base = String(o.baseUrl).replace(/\/+$/, '');
    var cdn = Object.assign({}, DEFAULT_CDN, o.cdn);
    var fetchFn = o.fetch || (root.fetch && root.fetch.bind(root));
    var AudioCtor = o.Audio || root.Audio;
    var loadScript = o.loadScript || loadScriptDefault;

    function trace() {
      if (!o.debug || !root.console) return;
      var a = Array.prototype.slice.call(arguments);
      a.unshift('[AriaClient]');
      root.console.log.apply(root.console, a);
    }

    var handlers = {};
    var state = 'idle';
    var listening = false;
    var muted = false;
    var destroyed = false;
    var abortCtl = null;
    var vadInst = null;
    var queue = [];
    var playing = false;
    var current = null;
    var ducked = false;
    var seen = {};
    var framesSeen = 0;
    var peak = 0;
    var lastTrace = 0;

    function on(name, fn) { (handlers[name] = handlers[name] || []).push(fn); return api; }
    function off(name, fn) { handlers[name] = (handlers[name] || []).filter(function (f) { return f !== fn; }); return api; }
    function emit(name, payload) {
      (handlers[name] || []).slice().forEach(function (fn) {
        try { fn(payload); } catch (e) { if (root.console) root.console.error('[AriaClient] gestionnaire « ' + name + ' » :', e); }
      });
    }
    function setState(s) {
      if (s === state || destroyed) return;
      trace('état :', state, '->', s);
      state = s;
      emit('state', s);
    }
    function idleState() { return listening ? 'listening' : 'idle'; }
    function headers() { return { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + o.apiKey }; }
    function fail(code, message) {
      emit('error', { code: code, message: message });
      setState(idleState());
    }

    // ---------------------------------------------------------------- lecture des blocs audio
    function playNext() {
      if (destroyed) return;
      if (muted || !queue.length) {
        playing = false;
        current = null;
        if (!queue.length) {
          emit('audio-end');
          if (!abortCtl) setState(idleState());
        }
        return;
      }
      playing = true;
      var chunk = queue.shift();
      var audio = new AudioCtor('data:audio/' + (chunk.format || 'wav') + ';base64,' + chunk.data);
      audio.volume = ducked ? 0.08 : 1;
      current = audio;
      setState('speaking');
      emit('audio-start', { id: chunk.id, text: chunk.text || '' });
      audio.onended = playNext;
      audio.onerror = playNext;
      var p = audio.play();
      if (p && p.catch) p.catch(function () { playNext(); });         // lecture bloquée (pas de geste utilisateur) : on passe au bloc suivant
    }
    function enqueue(chunk) {
      if (muted || !chunk || !chunk.data || seen[chunk.id]) return;
      seen[chunk.id] = true;
      queue.push(chunk);
      if (!playing) playNext();
    }
    function stopPlayback() {
      queue = [];
      playing = false;
      try { if (current && current.pause) current.pause(); } catch (e) { /* sans effet */ }
      current = null;
      ducked = false;
    }
    function duck() { ducked = true; try { if (current) current.volume = 0.08; } catch (e) { /* sans effet */ } }
    function unduck() { ducked = false; try { if (current) current.volume = 1; } catch (e) { /* sans effet */ } }

    // ---------------------------------------------------------------- un tour (POST + flux SSE)
    function handleEvent(ev) {
      var d = ev.data || {};
      if (ev.event === 'text') emit('text', d.text);
      else if (ev.event === 'user_text') emit('user_text', d.text);
      else if (ev.event === 'tool') emit('tool', d.name);
      else if (ev.event === 'audio') enqueue(d);
      else if (ev.event === 'done') emit('done', d);
      else if (ev.event === 'error') fail(d.code || 'error', d.message || 'erreur');
      else if (ev.event === 'state' && d.state === 'thinking' && !playing) setState('thinking');
    }

    async function turn(body) {
      if (destroyed) return;
      if (abortCtl) interrupt();                                       // un seul tour à la fois : le nouveau remplace l'ancien
      var ctl = new AbortController();
      abortCtl = ctl;
      seen = {};
      setState('thinking');
      var res;
      try {
        res = await fetchFn(base + '/v1/turn', {
          method: 'POST', headers: headers(), signal: ctl.signal,
          body: JSON.stringify(Object.assign({ session: o.session, speak: o.speak && !muted }, body))
        });
      } catch (e) {
        if (abortCtl === ctl) abortCtl = null;
        if (!ctl.signal.aborted) fail('network', String(e && e.message || e));
        return;
      }
      if (!res.ok) {
        var err = { code: 'http_' + res.status, message: res.statusText || 'erreur' };
        try { var j = await res.json(); if (j && j.error) err = j.error; } catch (e) { /* corps illisible */ }
        if (abortCtl === ctl) abortCtl = null;
        fail(err.code, err.message);
        return;
      }
      var reader = res.body.getReader();
      var dec = new TextDecoder();
      var buf = '';
      try {
        for (;;) {
          var r = await reader.read();
          if (r.done) break;
          buf += dec.decode(r.value, { stream: true });
          var p = parseSse(buf);
          buf = p.rest;
          p.events.forEach(handleEvent);
        }
      } catch (e) {
        if (!ctl.signal.aborted) fail('stream', String(e && e.message || e));
      }
      if (abortCtl === ctl) {
        abortCtl = null;
        if (!playing && !queue.length) setState(idleState());
      }
    }

    function say(text) {
      text = String(text == null ? '' : text).trim();
      if (!text) return Promise.resolve();
      return turn({ text: text });
    }

    function postJson(path) {
      return fetchFn(base + path, { method: 'POST', headers: headers(), body: JSON.stringify({ session: o.session }) })
        .catch(function () { /* meilleur effort */ });
    }

    function interrupt() {
      if (abortCtl) { abortCtl.abort(); abortCtl = null; postJson('/v1/stop'); }
      stopPlayback();
      setState(idleState());
    }

    function reset() { return postJson('/v1/reset'); }

    function mute(value) {
      muted = value !== false;
      if (muted) stopPlayback();
      return api;
    }

    // ---------------------------------------------------------------- écoute du micro (VAD dans le navigateur)
    async function start() {
      if (!o.voice) throw new Error('AriaClient : la voix est désactivée (option voice)');
      if (destroyed) throw new Error('AriaClient : client détruit');
      if (vadInst) { vadInst.start(); listening = true; setState('listening'); return; }
      trace('chargement du VAD (ort, vad-web) ; modèle :', o.modelUrl || (base + '/assets/silero_vad.onnx'));
      if (typeof root.ort === 'undefined') await loadScript(cdn.ort);
      if (typeof root.vad === 'undefined') await loadScript(cdn.vad);
      if (root.ort && root.ort.env) root.ort.env.logLevel = 'error';
      var v = o.vad || {};
      vadInst = await root.vad.MicVAD.new({
        modelURL: o.modelUrl || (base + '/assets/silero_vad.onnx'),
        positiveSpeechThreshold: v.positive != null ? v.positive : 0.8,
        negativeSpeechThreshold: v.negative != null ? v.negative : 0.4,
        redemptionFrames: v.redemption != null ? v.redemption : 20,
        minSpeechFrames: v.minFrames != null ? v.minFrames : 3,
        onFrameProcessed: o.debug ? function (probs) {
          // une trace par seconde : probabilité de parole MAXIMALE de la période (0 à 100 %) ; le seuil de détection est affiché en tête
          framesSeen++;
          peak = Math.max(peak, probs.isSpeech || 0);
          var now = Date.now();
          if (now - lastTrace >= 1000) {
            trace('VAD : parole ' + Math.round(peak * 100) + ' % (max sur 1 s, seuil ' + Math.round((v.positive != null ? v.positive : 0.8) * 100) + ' %), ' +
                  framesSeen + ' trame(s) reçues depuis le début');
            lastTrace = now;
            peak = 0;
          }
        } : function () {},
        onSpeechStart: function () {
          trace('VAD : début de parole');
          if (playing) duck();                                         // il parle pendant qu'Aria parle : sa voix baisse
          setState('hearing');
        },
        onVADMisfire: function () { trace('VAD : parole trop courte (ignorée)'); unduck(); setState(abortCtl ? 'thinking' : playing ? 'speaking' : idleState()); },
        onSpeechEnd: function (audio) {
          trace('VAD : fin de parole,', (audio.length / 16000).toFixed(1), 's audio envoyées à Aria');
          unduck();
          stopPlayback();                                              // la parole de l'utilisateur coupe la réponse en cours
          var wav = root.vad.utils.encodeWAV(audio);
          turn({ audio: 'data:audio/wav;base64,' + root.vad.utils.arrayBufferToBase64(wav) });
        }
      });
      await vadInst.start();
      listening = true;
      trace('VAD prêt : micro ouvert, en attente de parole');
      setState('listening');
    }

    function stop() {
      listening = false;
      if (vadInst) { try { vadInst.pause(); } catch (e) { /* sans effet */ } }
      if (!abortCtl && !playing) setState('idle');
    }

    function destroy() {
      stop();
      interrupt();
      destroyed = true;
      handlers = {};
      if (vadInst && vadInst.destroy) { try { vadInst.destroy(); } catch (e) { /* sans effet */ } }
      vadInst = null;
    }

    var api = {
      on: on, off: off, start: start, stop: stop, say: say, interrupt: interrupt, reset: reset, mute: mute, destroy: destroy,
      get state() { return state; }
    };
    return api;
  }

  var AriaClient = { create: create, parseSse: parseSse, version: '1.0' };
  root.AriaClient = AriaClient;
  if (typeof module !== 'undefined' && module.exports) module.exports = AriaClient;
})(typeof window !== 'undefined' ? window : globalThis);
