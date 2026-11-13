// Tests du client JavaScript (static/aria-client.js) avec un faux fetch, un faux Audio et un faux VAD. Lancer : node tests/test_aria_client.js
const assert = require('assert');
const path = require('path');
const AriaClient = require(path.join(__dirname, '..', 'static', 'aria-client.js'));

let ko = 0;
const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const enc = new TextEncoder();

function sse(events) {
  return events.map(([n, d]) => `event: ${n}\ndata: ${JSON.stringify(d)}\n\n`).join('');
}
function streamOf(text, chunk = 7) {
  const bytes = enc.encode(text);
  let i = 0;
  return { getReader() { return { read: async () => (i >= bytes.length ? { done: true } : { done: false, value: bytes.slice(i, (i += chunk)) }) }; } };
}
function fakeFetch(routes, log) {
  return async (url, init) => {
    const u = url.replace('https://aria.local:7861', '');
    log.push({ url: u, init });
    const r = routes[u];
    if (!r) return { ok: false, status: 404, statusText: 'nf', json: async () => ({ error: { code: 'nf', message: 'x' } }) };
    return typeof r === 'function' ? r(init) : r;
  };
}
class FakeAudio {
  constructor(src) { this.src = src; this.volume = 1; FakeAudio.all.push(this); }
  play() { setTimeout(() => this.onended && this.onended(), 0); return Promise.resolve(); }
  pause() { this.paused = true; }
}
FakeAudio.all = [];
const tick = (ms = 20) => new Promise(r => setTimeout(r, ms));

function client(routes, extra = {}) {
  const log = [];
  FakeAudio.all = [];
  const c = AriaClient.create({ baseUrl: 'https://aria.local:7861/', apiKey: 'ak_test', fetch: fakeFetch(routes, log), Audio: FakeAudio, ...extra });
  const seen = { states: [], text: [], tools: [], errors: [], user: [], done: 0 };
  c.on('state', s => seen.states.push(s)); c.on('text', t => seen.text.push(t)); c.on('tool', t => seen.tools.push(t));
  c.on('error', e => seen.errors.push(e)); c.on('user_text', t => seen.user.push(t)); c.on('done', () => seen.done++);
  return { c, log, seen };
}

test('parseSse : événements complets, reste conservé, blocs illisibles ignorés', () => {
  const p = AriaClient.parseSse('event: text\ndata: {"text":"a"}\n\nevent: done\ndata: {"ms":1}\n\nevent: te');
  assert.deepStrictEqual(p.events.map(e => e.event), ['text', 'done']);
  assert.strictEqual(p.rest, 'event: te');
  assert.deepStrictEqual(AriaClient.parseSse('event: x\ndata: pas json\n\n').events, []);
  assert.deepStrictEqual(AriaClient.parseSse('event: a\r\ndata: {"n":1}\r\n\r\n').events[0].data, { n: 1 });
});

test('création : baseUrl et apiKey obligatoires', () => {
  assert.throws(() => AriaClient.create({ apiKey: 'x' }));
  assert.throws(() => AriaClient.create({ baseUrl: 'x' }));
});

test('say : envoie la clé, le texte et la session, et reçoit texte, outil et fin', async () => {
  const body = sse([['state', { state: 'thinking' }], ['text', { text: 'Bon' }], ['tool', { name: 'meteo' }], ['text', { text: 'Bonjour' }], ['done', { ms: 5 }]]);
  const { c, log, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(body) } }, { session: 's1' });
  await c.say('  salut  ');
  const req = log[0];
  assert.strictEqual(req.url, '/v1/turn');
  assert.strictEqual(req.init.headers.Authorization, 'Bearer ak_test');
  assert.deepStrictEqual(JSON.parse(req.init.body), { session: 's1', speak: true, text: 'salut' });
  assert.deepStrictEqual(seen.text, ['Bon', 'Bonjour']);
  assert.deepStrictEqual(seen.tools, ['meteo']);
  assert.strictEqual(seen.done, 1);
  assert.strictEqual(c.state, 'idle');
  assert.deepStrictEqual(seen.states, ['thinking', 'idle']);
});

test('say : texte vide ignoré', async () => {
  const { c, log } = client({});
  await c.say('   ');
  assert.strictEqual(log.length, 0);
});

test('lecture : les blocs audio sont joués dans l’ordre, une seule fois, avec les bons événements', async () => {
  const wav = Buffer.from('RIFFx').toString('base64');
  const body = sse([['audio', { id: 'a1', format: 'wav', data: wav, text: 'Un' }], ['audio', { id: 'a1', format: 'wav', data: wav }],
                    ['audio', { id: 'a2', format: 'wav', data: wav, text: 'Deux' }], ['done', {}]]);
  const { c, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(body, 11) } });
  const ev = [];
  c.on('audio-start', e => ev.push('start:' + e.id)); c.on('audio-end', () => ev.push('end'));
  await c.say('x');
  await tick(60);
  assert.deepStrictEqual(ev, ['start:a1', 'start:a2', 'end']);
  assert.strictEqual(FakeAudio.all.length, 2);                                   // le doublon a1 est écarté
  assert.ok(FakeAudio.all[0].src.startsWith('data:audio/wav;base64,'));
  assert.ok(seen.states.includes('speaking'));
  assert.strictEqual(c.state, 'idle');
});

test('mute : aucun bloc joué, et speak=false demandé au serveur', async () => {
  const wav = Buffer.from('RIFFx').toString('base64');
  const { c, log } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['audio', { id: 'a1', data: wav }], ['done', {}]])) } });
  c.mute(true);
  await c.say('x');
  await tick(30);
  assert.strictEqual(FakeAudio.all.length, 0);
  assert.strictEqual(JSON.parse(log[0].init.body).speak, false);
});

test('erreurs HTTP : le code du serveur est relayé, rien ne plante', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: false, status: 429, statusText: 'x', json: async () => ({ error: { code: 'quota', message: 'quota quotidien atteint' } }) } });
  await c.say('x');
  assert.deepStrictEqual(seen.errors, [{ code: 'quota', message: 'quota quotidien atteint' }]);
  assert.strictEqual(c.state, 'idle');
});

test('erreur réseau', async () => {
  const log = [];
  const c = AriaClient.create({ baseUrl: 'https://aria.local:7861', apiKey: 'k', fetch: async () => { throw new Error('boom'); }, Audio: FakeAudio });
  const errs = [];
  c.on('error', e => errs.push(e));
  await c.say('x');
  assert.strictEqual(errs[0].code, 'network');
  assert.strictEqual(c.state, 'idle');
});

test('événement error du flux', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['error', { code: 'llm', message: 'le modèle n’a pas pu répondre' }]])) } });
  await c.say('x');
  assert.strictEqual(seen.errors[0].code, 'llm');
});

test('tour vocal : user_text relayé', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['user_text', { text: 'quelle heure' }], ['done', {}]])) } });
  await c.say('x');
  assert.deepStrictEqual(seen.user, ['quelle heure']);
});

test('interrupt : coupe la lecture, prévient le serveur', async () => {
  const wav = Buffer.from('RIFFx').toString('base64');
  let release;
  const gate = new Promise(r => { release = r; });
  const slow = { getReader() { let sent = false; return { read: async () => { if (!sent) { sent = true; return { done: false, value: enc.encode(sse([['audio', { id: 'a1', data: wav }]])) }; } await gate; return { done: true }; } }; } };
  const { c, log } = client({ '/v1/turn': { ok: true, status: 200, body: slow }, '/v1/stop': { ok: true, status: 200, json: async () => ({ ok: true }) } });
  const p = c.say('x');
  await tick(10);
  c.interrupt();
  release();
  await p;
  assert.ok(log.some(l => l.url === '/v1/stop'));
  assert.strictEqual(c.state, 'idle');
});

test('reset et destroy', async () => {
  const { c, log } = client({ '/v1/reset': { ok: true, status: 200, json: async () => ({}) } });
  await c.reset();
  assert.ok(log.some(l => l.url === '/v1/reset'));
  c.destroy();
  await c.say('x');
  assert.strictEqual(log.filter(l => l.url === '/v1/turn').length, 0);
});

test('voix désactivée : start refuse', async () => {
  const { c } = client({}, { voice: false });
  await assert.rejects(() => c.start());
});

test('start : charge ort puis vad, crée le VAD avec le modèle d’Aria, envoie l’audio détecté', async () => {
  const loaded = [];
  const wavBuf = new ArrayBuffer(8);
  const g = globalThis;
  let created = null;
  g.ort = undefined; g.vad = undefined;
  const loadScript = async src => {
    loaded.push(src);
    if (src.includes('ort')) g.ort = { env: {} };
    else g.vad = {
      MicVAD: { new: async opts => { created = opts; return { start: async () => {}, pause() {}, destroy() {} }; } },
      utils: { encodeWAV: () => wavBuf, arrayBufferToBase64: () => 'QUJD' }
    };
  };
  const { c, log, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['user_text', { text: 'salut' }], ['done', {}]])) } }, { loadScript });
  await c.start();
  assert.strictEqual(loaded.length, 2);
  assert.strictEqual(created.modelURL, 'https://aria.local:7861/assets/silero_vad.onnx');
  assert.strictEqual(c.state, 'listening');
  created.onSpeechStart();
  assert.strictEqual(c.state, 'hearing');
  created.onSpeechEnd(new Float32Array(10));
  await tick(40);
  const sent = JSON.parse(log.find(l => l.url === '/v1/turn').init.body);
  assert.strictEqual(sent.audio, 'data:audio/wav;base64,QUJD');
  assert.deepStrictEqual(seen.user, ['salut']);
  assert.strictEqual(c.state, 'listening');
  c.stop();
  assert.strictEqual(c.state, 'idle');
  await c.start();
  assert.strictEqual(loaded.length, 2);                                          // le VAD existant est relancé, pas recréé
  delete g.ort; delete g.vad;
});

// ---------------------------------------------------------------- prompt système de l'application
test('systemPrompt : envoyé dans le champ system, nettoyé, à chaque tour', async () => {
  const { c, log } = client({ '/v1/turn': () => ({ ok: true, status: 200, body: streamOf(sse([['done', {}]])) }) }, { systemPrompt: '  Tu es l’assistante du site Bleu.  ' });
  await c.say('un');
  await c.say('deux');
  const bodies = log.filter(l => l.url === '/v1/turn').map(l => JSON.parse(l.init.body));
  assert.strictEqual(bodies.length, 2);
  assert.ok(bodies.every(b => b.system === 'Tu es l’assistante du site Bleu.'));
});

test('systemPrompt : absent ou vide => pas de champ system', async () => {
  for (const extra of [{}, { systemPrompt: '' }, { systemPrompt: '   ' }]) {
    const { c, log } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['done', {}]])) } }, extra);
    await c.say('x');
    assert.ok(!('system' in JSON.parse(log[0].init.body)), JSON.stringify(extra));
  }
});

test('systemPrompt : un non-texte est refusé à la création', () => {
  assert.throws(() => AriaClient.create({ baseUrl: 'https://a', apiKey: 'k', systemPrompt: 42 }), /texte/);
  assert.throws(() => AriaClient.create({ baseUrl: 'https://a', apiKey: 'k', systemPrompt: { a: 1 } }), /texte/);
});

test('systemPrompt : accompagne aussi les tours vocaux', async () => {
  const g = globalThis; g.ort = undefined; g.vad = undefined;
  let created = null;
  const loadScript = async src => {
    if (src.includes('ort')) g.ort = { env: {} };
    else g.vad = { MicVAD: { new: async o => { created = o; return { start: async () => {}, pause() {}, destroy() {} }; } }, utils: { encodeWAV: () => new ArrayBuffer(4), arrayBufferToBase64: () => 'QQ==' } };
  };
  const { c, log } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['done', {}]])) } }, { loadScript, systemPrompt: 'Consignes' });
  await c.start();
  created.onSpeechEnd(new Float32Array(4));
  await tick(30);
  const b = JSON.parse(log.find(l => l.url === '/v1/turn').init.body);
  assert.strictEqual(b.system, 'Consignes');
  assert.ok(b.audio);
  delete g.ort; delete g.vad;
});

test('avertissement du serveur relayé en événement warning, sans être une erreur', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['warning', { code: 'system_ignored', message: 'déjà créée' }], ['done', {}]])) } });
  const warns = [];
  c.on('warning', w => warns.push(w));
  await c.say('x');
  assert.deepStrictEqual(warns, [{ code: 'system_ignored', message: 'déjà créée' }]);
  assert.strictEqual(seen.errors.length, 0);
  assert.strictEqual(seen.done, 1);
});

test('erreur 413 sur le prompt : relayée, le client reste utilisable', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: false, status: 413, statusText: 'x', json: async () => ({ error: { code: 'too_large', message: 'system : 4000 caractères au plus' } }) } },
    { systemPrompt: 'x'.repeat(5000) });
  await c.say('x');
  assert.strictEqual(seen.errors[0].code, 'too_large');
  assert.strictEqual(c.state, 'idle');
});

// ---------------------------------------------------------------- visualisation : niveaux, analyseur, liaison CSS
class FakeAnalyser {
  constructor(level) { this.level = level; this.fftSize = 1024; this.smoothingTimeConstant = 0; this.connected = []; }
  connect(n) { this.connected.push(n); return n; }
  getByteTimeDomainData(buf) { for (let i = 0; i < buf.length; i++) buf[i] = 128 + Math.round((i % 2 ? 1 : -1) * this.level * 128); }
}
function fakeAudioContext(levels) {
  const made = [];
  class AC {
    constructor() { this.state = 'running'; this.destination = { dest: true }; this.sources = []; AC.last = this; }
    createAnalyser() { const a = new FakeAnalyser(levels.next ? levels.next() : levels.value); made.push(a); return a; }
    createMediaElementSource(el) { const n = { el, connect(t) { this.to = t; } }; this.sources.push(n); return n; }
    createMediaStreamSource(st) { const n = { st, connect(t) { this.to = t; } }; this.sources.push(n); return n; }
    resume() {} close() { this.closed = true; }
  }
  AC.made = made;
  return AC;
}
class FakeEl {
  constructor() { this.attrs = {}; this.props = {}; this.style = { setProperty: (k, v) => { this.props[k] = v; }, removeProperty: k => { delete this.props[k]; } }; }
  setAttribute(k, v) { this.attrs[k] = v; } removeAttribute(k) { delete this.attrs[k]; }
}
const WAV = Buffer.from('RIFFx').toString('base64');
const voiceBody = () => streamOf(sse([['audio', { id: 'a1', data: WAV, text: 'Un' }], ['done', {}]]));

test('level : sans abonné, aucune analyse n’est créée', async () => {
  const AC = fakeAudioContext({ value: 0.3 });
  const { c } = client({ '/v1/turn': { ok: true, status: 200, body: voiceBody() } }, { AudioContext: AC });
  await c.say('x'); await tick(40);
  assert.strictEqual(AC.made.length, 0);
});

test('level : abonné => la voix d’Aria passe par un analyseur relié aux haut-parleurs, et le niveau est émis', async () => {
  const AC = fakeAudioContext({ value: 0.2 });
  class LongAudio extends FakeAudio { play() { setTimeout(() => this.onended && this.onended(), 220); return Promise.resolve(); } }
  const { c } = client({ '/v1/turn': { ok: true, status: 200, body: voiceBody() } }, { AudioContext: AC, Audio: LongAudio });
  const levels = [];
  c.on('level', l => levels.push(l));
  await c.say('x'); await tick(160);
  assert.strictEqual(AC.made.length, 1);
  assert.deepStrictEqual(AC.made[0].connected, [AC.last.destination]);             // le son reste audible
  assert.strictEqual(AC.last.sources.length, 1);
  assert.strictEqual(AC.last.sources[0].to, AC.made[0]);
  const voice = levels.filter(l => l.source === 'voice');
  assert.ok(voice.length >= 2, 'plusieurs mesures pendant la parole');
  assert.ok(voice.every(l => l.value > 0 && l.value <= 1), 'niveaux entre 0 et 1');
  await tick(200);
  assert.strictEqual(levels.filter(l => l.source === 'voice').pop().value, 0);       // un dernier 0 à la fin de la parole
  c.destroy();
  assert.ok(AC.last.closed);
});

test('audio-start donne l’élément audio et le texte du bloc', async () => {
  const { c } = client({ '/v1/turn': { ok: true, status: 200, body: voiceBody() } });
  let got = null;
  c.on('audio-start', e => { got = e; });
  await c.say('x'); await tick(30);
  assert.strictEqual(got.text, 'Un');
  assert.ok(got.element instanceof FakeAudio);
});

test('analyser("voice") : même instance à chaque appel ; "mic" : null tant que le micro n’est pas ouvert', async () => {
  const AC = fakeAudioContext({ value: 0.1 });
  const { c } = client({}, { AudioContext: AC });
  const a = c.analyser('voice');
  assert.ok(a && typeof a.getByteTimeDomainData === 'function');
  assert.strictEqual(c.analyser('voice'), a);
  assert.strictEqual(c.analyser('mic'), null);
  c.destroy();
});

test('analyser("mic") après start : relié au flux du micro, pas aux haut-parleurs', async () => {
  const AC = fakeAudioContext({ value: 0.1 });
  const g = globalThis; g.ort = undefined; g.vad = undefined;
  const stream = { id: 'flux' };
  const loadScript = async src => {
    if (src.includes('ort')) g.ort = { env: {} };
    else g.vad = { MicVAD: { new: async () => ({ stream, start: async () => {}, pause() {}, destroy() {} }) }, utils: {} };
  };
  const { c } = client({}, { AudioContext: AC, loadScript });
  await c.start();
  const mic = c.analyser('mic');
  assert.ok(mic);
  assert.deepStrictEqual(mic.connected, []);
  assert.strictEqual(AC.last.sources[0].st, stream);
  assert.strictEqual(c.listening, true);
  delete g.ort; delete g.vad;
});

test('bindState : data-aria-state et --aria-level sur l’élément, et retrait propre', async () => {
  const AC = fakeAudioContext({ value: 0.2 });
  const { c } = client({ '/v1/turn': { ok: true, status: 200, body: voiceBody() } }, { AudioContext: AC });
  const el = new FakeEl();
  const unbind = c.bindState(el);
  assert.strictEqual(el.attrs['data-aria-state'], 'idle');
  await c.say('x'); await tick(40);
  assert.ok(['thinking', 'speaking', 'idle'].includes(el.attrs['data-aria-state']));
  unbind();
  assert.strictEqual(el.attrs['data-aria-state'], undefined);
  assert.strictEqual(el.props['--aria-level'], undefined);
  c.destroy();
});

test('bindState suit les états', async () => {
  const { c } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['text', { text: 'a' }], ['done', {}]])) } });
  const el = new FakeEl();
  const seen = [];
  c.bindState(el, { level: false });
  c.on('state', () => seen.push(el.attrs['data-aria-state']));
  await c.say('x');
  assert.deepStrictEqual(seen, ['thinking', 'idle']);
  assert.strictEqual(el.attrs['data-aria-state'], 'idle');
});

test('un gestionnaire qui lève ne casse pas le client', async () => {
  const { c, seen } = client({ '/v1/turn': { ok: true, status: 200, body: streamOf(sse([['text', { text: 'a' }], ['done', {}]])) } });
  c.on('text', () => { throw new Error('page tierce boguée'); });
  const orig = console.error; console.error = () => {};
  await c.say('x');
  console.error = orig;
  assert.strictEqual(seen.done, 1);
});

test('mute / muted / listening exposés', () => {
  const { c } = client({});
  assert.strictEqual(c.muted, false);
  c.mute(true);
  assert.strictEqual(c.muted, true);
  assert.strictEqual(c.listening, false);
});

(async () => {
  for (const [name, fn] of tests) {
    try { await fn(); console.log('ok  ', name); } catch (e) { ko++; console.log('KO  ', name, '\n     ', e.message); }
  }
  console.log(ko ? `\n${ko} test(s) en échec` : `\n${tests.length} tests OK`);
  process.exit(ko ? 1 : 0);
})();
