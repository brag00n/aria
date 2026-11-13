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

(async () => {
  for (const [name, fn] of tests) {
    try { await fn(); console.log('ok  ', name); } catch (e) { ko++; console.log('KO  ', name, '\n     ', e.message); }
  }
  console.log(ko ? `\n${ko} test(s) en échec` : `\n${tests.length} tests OK`);
  process.exit(ko ? 1 : 0);
})();
