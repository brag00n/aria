"""Page de visualisation et de gestion des souvenirs (bibliothèque standard, aucune dépendance, aucun appel externe).

Démarrée par server.py seulement si MEMORY_UI_PORT est défini, dans un thread du même processus : un seul écrivain, la page
voit toujours l'index à jour et passe par les MÊMES fonctions que les outils MCP (store.add / update / delete / restore).

Sécurité (la page peut afficher du texte venu du web via la mémoire) :
  - tout texte est inséré dans le DOM avec textContent (jamais innerHTML) ;
  - l'en-tête Host doit correspondre à une adresse locale ou à MEMORY_UI_HOSTS (anti DNS rebinding) ;
  - les POST exigent un en-tête Origin identique au Host (anti CSRF) et du JSON ;
  - corps limité à 16 Ko ; jeton optionnel MEMORY_UI_TOKEN (en-tête X-Token) ;
  - écoute sur 0.0.0.0 dans le conteneur ; c'est la publication du port par docker-compose qui la limite à 127.0.0.1.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_BODY = 16 * 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def make_handler(store, activity, token):
    from store import iso   # importé ici : ui.py reste importable même si store n'est pas encore chargé

    allowed_hosts = set(h.strip().lower() for h in os.environ.get("MEMORY_UI_HOSTS", "").split(",") if h.strip())

    def view(rec, now):
        return {"id": rec["id"], "type": rec["t"], "texte": rec["txt"], "imp": rec["imp"], "s": round(rec["s"], 2),
                "eff": round(store.eff(rec, now), 2), "tags": rec["tags"], "etat": store.state(rec, now),
                "cree": iso(rec["c"]), "dernier": iso(rec["last"]), "hits": rec["hits"], "src": rec["src"]}

    class Handler(BaseHTTPRequestHandler):
        server_version = "MemoireUI"
        sys_version = ""

        def log_message(self, fmt, *args):      # pas de bruit sur stderr
            pass

        # ---- utilitaires
        def _host_ok(self):
            host = (self.headers.get("Host") or "").lower()
            name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
            return name in LOCAL_HOSTS or host in allowed_hosts or name in allowed_hosts

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False))

        def _guard(self, post):
            if not self._host_ok():
                self._json({"erreur": "hôte non autorisé"}, 403)
                return False
            if token and self.headers.get("X-Token") != token and not self.path.startswith("/?"):
                if self.path != "/" and not self.path.startswith("/static"):
                    self._json({"erreur": "jeton requis"}, 401)
                    return False
            if post:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
                    self._json({"erreur": "origine refusée"}, 403)
                    return False
                if "application/json" not in (self.headers.get("Content-Type") or ""):
                    self._json({"erreur": "JSON attendu"}, 415)
                    return False
            return True

        def _body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if n <= 0 or n > MAX_BODY:
                return None
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
                return obj if isinstance(obj, dict) else None
            except (ValueError, UnicodeDecodeError):
                return None

        # ---- routes
        def do_GET(self):
            if not self._guard(False):
                return
            url = urlparse(self.path)
            q = parse_qs(url.query)
            now = store.clock()
            if url.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if url.path == "/api/memories":
                text = (q.get("q") or [""])[0]
                if text.strip():
                    recs, _ = store.search(text, 10, touch=False, src="ui")
                else:
                    recs = store.all_records(now)
                return self._json([view(r, now) for r in recs])
            if url.path == "/api/stats":
                return self._json(store.stats())
            if url.path == "/api/archive":
                return self._json(store.archived())
            if url.path == "/api/log":
                return self._json(activity.tail(200))
            self._json({"erreur": "introuvable"}, 404)

        def do_POST(self):
            if not self._guard(True):
                return
            body = self._body()
            if body is None:
                return self._json({"erreur": "corps invalide ou trop grand"}, 400)
            path = urlparse(self.path).path
            activity.log("ui", action=path.rsplit("/", 1)[-1])
            if path == "/api/add":
                status, rec = store.add(str(body.get("texte", "")), str(body.get("type", "fait")),
                                        body.get("tags", ""), src="ui")
                return self._json({"statut": status, "id": rec["id"] if rec else None})
            rid = str(body.get("id", ""))
            if path == "/api/update":
                changes = {}
                if "texte" in body: changes["txt"] = str(body["texte"])
                if "type" in body: changes["t"] = str(body["type"])
                if "tags" in body: changes["tags"] = body["tags"]
                if "imp" in body: changes["imp"] = body["imp"]
                return self._json({"ok": store.update(rid, changes, src="ui")})
            if path == "/api/delete":
                return self._json({"ok": store.delete(rid, why="ui", src="ui")})
            if path == "/api/restore":
                return self._json({"ok": store.restore(rid, src="ui")})
            if path == "/api/pin":
                rec = store.records.get(rid)
                if not rec:
                    return self._json({"ok": False})
                tags = [t for t in rec["tags"] if t != "pinned"]
                if "pinned" not in rec["tags"]:
                    tags.append("pinned")
                return self._json({"ok": store.update(rid, {"tags": tags}, src="ui")})
            self._json({"erreur": "introuvable"}, 404)

    return Handler


def start(store, activity, port, token=None):
    """Lance la page dans un thread démon. Une erreur ici ne doit jamais empêcher le serveur de mémoire de fonctionner."""
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store, activity, token))
    thread = threading.Thread(target=httpd.serve_forever, name="memoire-ui", daemon=True)
    thread.start()
    activity.log("ui_start", port=port, token=bool(token))
    activity.warn(f"page de gestion sur le port {port}")
    return httpd


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mémoire d'Aria</title>
<style>
:root{--bg:#fafafa;--fg:#1c1c1e;--mut:#6b6b70;--line:#d9d9de;--card:#fff;--acc:#2a6df4;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#33333a;--card:#1f1f23;--acc:#6c9bff;--warn:#f0a35a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}nav button{margin-right:4px}
main{padding:12px 16px}
button,input,select{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{border-color:var(--acc);color:var(--acc)}
table{width:100%;border-collapse:collapse;background:var(--card)}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{cursor:pointer;color:var(--mut);font-weight:600;white-space:nowrap}
.mut{color:var(--mut)}.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 6px;margin-right:3px;font-size:12px}
.masque{opacity:.55}.protege td:first-child{border-left:3px solid var(--acc)}
.bar{display:flex;gap:8px;margin:8px 0;flex-wrap:wrap}
pre{background:var(--card);border:1px solid var(--line);padding:8px;overflow:auto;max-height:60vh}
#msg{color:var(--warn);margin-left:8px}
</style></head><body>
<header><h1>🧠 Mémoire d'Aria</h1>
<nav><button data-t="mem" class="on">Souvenirs</button><button data-t="arc">Archive</button><button data-t="sta">Statistiques</button><button data-t="log">Journal</button></nav>
<span id="msg"></span></header>
<main id="main"></main>
<script>
const $=(s,r=document)=>r.querySelector(s);
const el=(tag,props={},...kids)=>{const e=document.createElement(tag);Object.assign(e,props);kids.flat().forEach(k=>e.append(k));return e};
const TOKEN=new URLSearchParams(location.search).get('token')||'';
const hdr=()=>Object.assign({'Content-Type':'application/json'},TOKEN?{'X-Token':TOKEN}:{});
const get=async u=>(await fetch(u,{headers:hdr()})).json();
const post=async(u,b)=>(await fetch(u,{method:'POST',headers:hdr(),body:JSON.stringify(b)})).json();
const say=t=>{$('#msg').textContent=t;setTimeout(()=>{$('#msg').textContent=''},4000)};
let tab='mem',sortKey='cree',sortDir=-1,query='',fType='',fState='';

async function render(){
  const main=$('#main');main.replaceChildren();
  if(tab==='mem')return renderMem(main);
  if(tab==='arc'){const rows=await get('/api/archive');
    return main.append(el('table',{},el('tr',{},...['Texte','Type','Raison','Date','Id',''].map(h=>el('th',{},h))),
      ...rows.map(r=>el('tr',{},el('td',{},r.txt),el('td',{},r.t),el('td',{},r.why),el('td',{className:'mut'},r.ts),el('td',{className:'mut'},r.id),
        el('td',{},el('button',{onclick:async()=>{say((await post('/api/restore',{id:r.id})).ok?'Restauré':'Impossible');render()}},'Restaurer'))))));}
  if(tab==='sta'){const s=await get('/api/stats');return main.append(el('pre',{},JSON.stringify(s,null,2)));}
  if(tab==='log'){const l=await get('/api/log');return main.append(el('pre',{},l.map(x=>JSON.stringify(x)).join('\\n')));}
}
async function renderMem(main){
  const rows=await get('/api/memories?q='+encodeURIComponent(query));
  let list=rows.filter(r=>(!fType||r.type===fType)&&(!fState||r.etat===fState));
  list.sort((a,b)=>(a[sortKey]>b[sortKey]?1:-1)*sortDir);
  const inpQ=el('input',{placeholder:'Rechercher…',value:query,onchange:e=>{query=e.target.value;render()}});
  const selT=el('select',{onchange:e=>{fType=e.target.value;render()}},...['','fait','preference','tache','ephemere'].map(v=>el('option',{value:v,selected:v===fType},v||'tous les types')));
  const selS=el('select',{onchange:e=>{fState=e.target.value;render()}},...['','actif','masque','protege'].map(v=>el('option',{value:v,selected:v===fState},v||'tous les états')));
  const nt=el('input',{placeholder:'Nouveau souvenir…',size:40}),ntype=el('select',{},...['fait','preference','tache','ephemere'].map(v=>el('option',{value:v},v))),ntags=el('input',{placeholder:'tags',size:14});
  const add=el('button',{onclick:async()=>{if(!nt.value.trim())return;const r=await post('/api/add',{texte:nt.value,type:ntype.value,tags:ntags.value});say(r.statut);render()}},'Ajouter');
  const th=(k,t)=>el('th',{onclick:()=>{sortDir=(sortKey===k)?-sortDir:-1;sortKey=k;render()}},t+(sortKey===k?(sortDir>0?' ▲':' ▼'):''));
  main.append(el('div',{className:'bar'},inpQ,selT,selS),el('div',{className:'bar'},nt,ntype,ntags,add),
    el('table',{},el('tr',{},th('texte','Souvenir'),th('type','Type'),th('eff','Importance'),th('tags','Tags'),th('cree','Créé'),th('hits','Rappels'),th('etat','État'),el('th',{},'')),
    ...list.map(r=>el('tr',{className:r.etat},
      el('td',{},r.texte,el('div',{className:'mut'},r.id+' · '+r.src)),el('td',{},r.type),
      el('td',{},String(r.eff),el('div',{className:'mut'},r.imp+' × '+r.s)),
      el('td',{},...r.tags.map(t=>el('span',{className:'tag'},t))),el('td',{className:'mut'},r.cree.slice(0,10)),el('td',{},String(r.hits)),el('td',{},r.etat),
      el('td',{},el('button',{onclick:async()=>{const t=prompt('Modifier le souvenir',r.texte);if(t&&t!==r.texte){await post('/api/update',{id:r.id,texte:t});render()}}},'✎'),' ',
        el('button',{title:'Protéger / déprotéger',onclick:async()=>{await post('/api/pin',{id:r.id});render()}},'📌'),' ',
        el('button',{title:'Oublier',onclick:async()=>{if(confirm('Oublier ce souvenir ?')){await post('/api/delete',{id:r.id});render()}}},'🗑'))))));
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));render()});
render();
</script></body></html>
"""
