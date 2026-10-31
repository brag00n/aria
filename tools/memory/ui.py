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

import liens
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
                "cree": iso(rec["c"]), "dernier": iso(rec["last"]), "hits": rec["hits"], "src": rec["src"],
                "o": rec.get("o", ""), "proprietaire": owner_label(rec.get("o", "")), "rel": rec.get("rel")}

    def owner_label(pid):
        """Libellé du propriétaire : « Olivier (Cousinier) », « Foyer » si le souvenir est partagé, l'identifiant si la personne a disparu."""
        if not pid:
            return "Foyer (partagé)"
        person = store.personnes.get(pid)
        return store.personnes.label(person) if person else f"? {pid}"

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
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass        # le navigateur a fermé la connexion (onglet fermé ou rechargé) : sans conséquence

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
                    recs, _ = store.search(text, 10, touch=False, src="ui", everything=True)
                else:
                    recs = store.all_records(now)
                return self._json([view(r, now) for r in recs])
            if url.path == "/api/personnes":
                counts = {}
                for r in store.records.values():
                    counts[r.get("o", "")] = counts.get(r.get("o", ""), 0) + 1
                return self._json([{"id": p["id"], "label": store.personnes.label(p), "nom": p["nom"], "compl": p["compl"],
                                    "surnom": p["surnom"], "sexe": p.get("sexe", ""), "citee": bool(p.get("ref")),
                                    "cree": p["c"], "souvenirs": counts.get(p["id"], 0),
                                    "actif": p["id"] == store.active} for p in store.personnes.items]
                                   + [{"id": "", "label": "Foyer (partagé)", "souvenirs": counts.get("", 0), "actif": False, "special": True}])
            if url.path == "/api/graphe":
                from interlocuteur import Interlocuteurs
                graphe = Interlocuteurs(store).graphe((q.get("id") or [""])[0], (q.get("niveau") or ["1"])[0])
                return self._json(graphe if graphe else {"erreur": "personne inconnue"}, 200 if graphe else 404)
            if url.path == "/api/relations":
                return self._json([view(r, now) for r in store.records.values() if r["t"] == "relation"])
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
                owner = str(body.get("o", "")) if body.get("o") else None
                if owner is not None and not store.personnes.get(owner):
                    return self._json({"statut": "personne_inconnue", "id": None})
                status, rec = store.add(str(body.get("texte", "")), str(body.get("type", "fait")),
                                        body.get("tags", ""), src="ui", owner=owner)      # sans propriétaire précisé : l'interlocuteur actif
                return self._json({"statut": status, "id": rec["id"] if rec else None})
            if path == "/api/relation/add":
                from interlocuteur import Interlocuteurs
                ok, msg = Interlocuteurs(store).relier_fiches(str(body.get("a", "")), str(body.get("lien", "")), str(body.get("b", "")),
                                                              str(body.get("precision", "")), src="ui")
                return self._json({"ok": ok, "id": msg if ok else None, "erreur": None if ok else msg})
            rid = str(body.get("id", ""))
            if path == "/api/update":
                changes = {}
                if "texte" in body: changes["txt"] = str(body["texte"])
                if "type" in body: changes["t"] = str(body["type"])
                if "tags" in body: changes["tags"] = body["tags"]
                if "imp" in body: changes["imp"] = body["imp"]
                if "s" in body: changes["s"] = body["s"]
                if isinstance(body.get("rel"), dict):
                    rec = store.records.get(rid)
                    if not rec or rec["t"] != "relation":
                        return self._json({"ok": False, "erreur": "ce souvenir n'est pas une relation"})
                    a, l, b = (" ".join(str(body["rel"].get(k, "")).split())[:60] for k in ("a", "l", "b"))
                    if not l and body["rel"].get("t"):
                        l = str(body["rel"]["t"])
                    if not (a and l and b) or a.lower() == b.lower():
                        return self._json({"ok": False, "erreur": "une relation a besoin de deux personnes différentes et d'un lien"})
                    t, l = liens.choisir(l, body["rel"].get("precision", ""))
                    if t is None:
                        return self._json({"ok": False, "erreur": "lien hors liste : " + ", ".join(liens.TYPES)})
                    info = liens.analyser(a, l, b, t)
                    changes["rel"] = {"a": a, "l": l, "b": b, "t": t, "cle": info["cle"], "inv": info["inv"],
                                      "ia": (store.personnes.find(a) or [{"id": ""}])[0]["id"] if len(store.personnes.find(a)) == 1 else "",
                                      "ib": (store.personnes.find(b) or [{"id": ""}])[0]["id"] if len(store.personnes.find(b)) == 1 else ""}
                    changes["txt"] = f"{a} est {liens.article(l)} {b}"
                if "o" in body:
                    if body["o"] and not store.personnes.get(str(body["o"])):
                        return self._json({"ok": False, "erreur": "personne inconnue"})
                    changes["o"] = str(body["o"] or "")
                return self._json({"ok": store.update(rid, changes, src="ui")})
            if path == "/api/delete":
                return self._json({"ok": store.delete(rid, why="ui", src="ui", force=True)})
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
            if path == "/api/personne/add":
                person = store.personnes.add(body.get("nom", ""), body.get("compl", ""), body.get("surnom", ""), body.get("sexe", ""))
                return self._json({"ok": bool(person), "id": person["id"] if person else None})
            if path == "/api/personne/update":
                fields = {k: body[k] for k in ("nom", "compl", "surnom", "sexe") if k in body}
                if "connue" in body:                      # connue d'Aria = a déjà discuté avec elle = présente dans la liste des interlocuteurs
                    if not body["connue"] and store.active == rid:
                        return self._json({"ok": False, "erreur": "cette personne est l'interlocuteur actif : choisis d'abord quelqu'un d'autre"})
                    fields["ref"] = not body["connue"]
                ok = store.personnes.update(rid, **fields)
                if ok and ("nom" in fields or "compl" in fields):
                    from interlocuteur import Interlocuteurs
                    Interlocuteurs(store).synchroniser_noms(rid)      # les relations suivent le nouveau nom
                return self._json({"ok": ok})
            if path == "/api/personne/merge":
                moved = store.merge_persons(rid, str(body.get("vers", "")), src="ui")
                if moved >= 0:
                    from interlocuteur import Interlocuteurs
                    Interlocuteurs(store).synchroniser_noms(str(body.get("vers", "")))      # les relations reprennent le nom de la personne gardée
                return self._json({"ok": moved >= 0, "deplaces": max(moved, 0)})
            if path == "/api/personne/delete":
                if any(r.get("o") == rid for r in store.records.values()):
                    return self._json({"ok": False, "erreur": "cette personne a encore des souvenirs : rattache-les ou fusionne d'abord"})
                if store.active == rid:
                    store.active = ""
                return self._json({"ok": store.personnes.remove(rid)})
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
:root{--fam:#2a6df4;--con:#e07b00;--ctr:#1c1c1e}
@media (prefers-color-scheme:dark){:root{--fam:#6c9bff;--con:#f0a35a;--ctr:#ececf0}}
#modal .wide{min-width:min(760px,92vw)}
#graph{width:100%;height:auto;max-height:70vh;min-height:300px;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
#graph .node{cursor:pointer}#graph .node:hover circle{stroke-width:4}
#graph text{font:12px system-ui,sans-serif;fill:var(--fg);pointer-events:none}
#graph .edge-label{font-size:11px;fill:var(--mut)}
.legend{display:flex;flex-wrap:wrap;align-items:center;gap:16px;margin:6px 0;font-size:12px;color:var(--mut)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px;vertical-align:-1px}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;z-index:10}
#modal .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;min-width:360px;max-width:92vw;max-height:90vh;overflow:auto}
#modal h2{font-size:15px;margin:0 0 10px}
#modal label{display:block;margin:8px 0 2px;color:var(--mut);font-size:12px}
#modal input,#modal select,#modal textarea{width:100%;box-sizing:border-box;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
#modal textarea{min-height:80px;resize:vertical}
#modal .row{display:flex;gap:10px}#modal .row>div{flex:1}
#modal .chk{display:flex;align-items:center;gap:6px;margin-top:10px}#modal .chk input{width:auto}
#modal .info{margin-top:10px;font-size:12px;color:var(--mut)}
#modal .btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
#modal .err{color:var(--warn);margin-top:8px;min-height:1em}
</style></head><body>
<header><h1>🧠 Mémoire d'Aria</h1>
<nav><button data-t="mem" class="on">Souvenirs</button><button data-t="per">Personnes</button><button data-t="rel">Relations</button><button data-t="arc">Archive</button><button data-t="sta">Statistiques</button><button data-t="log">Journal</button></nav>
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
let tab='mem',sortKey='cree',sortDir=-1,query='',pq='',rq='',pf={champ:'',sexe:'',statut:'',souv:''},rf={champ:'',pers:'',lien:'',cat:''},fType='',fState='',fOwner='',PERSONS=[];
const norm=t=>String(t||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase();
const match=(q,...champs)=>!norm(q).trim()||norm(q).trim().split(/\s+/).every(w=>champs.some(c=>norm(c).includes(w)));
const listbox=(etat,cle,choix)=>el('select',{title:cle,onchange:e=>{etat[cle]=e.target.value;render()}},...choix.map(c=>el('option',{value:c[0],selected:c[0]===etat[cle]},c[1])));
const CAT_LIEN={conjoint:'famille',parent:'famille',enfant:'famille',fratrie:'famille',demi:'famille',beauparent:'famille',beauenfant:'famille'};
const catDe=r=>CAT_LIEN[(r.rel&&r.rel.t)||'autre']||'connaissance';
const OWNER_TXT=id=>(PERSONS.find(p=>p.id===id)||{label:id?'? '+id:'Foyer (partagé)'}).label;

// ---- Modale d'édition : tous les champs modifiables d'un souvenir, d'une relation ou d'une personne
function field(label, node){ return el('div',{}, el('label',{},label), node); }
function openModal(title, fields, info, onSave, retour){
  closeModal();
  const err = el('div',{className:'err'});
  const save = el('button',{onclick:async()=>{ err.textContent=''; const res = await onSave(); if(res===true){closeModal();render()} else err.textContent = res || 'Impossible'; }},'Enregistrer');
  const cancel = el('button',{onclick:closeModal},'Annuler');
  const box = el('div',{id:'modal',onclick:e=>{if(e.target===box)closeModal()}},
    el('div',{className:'card'}, el('h2',{},title), ...fields, info?el('div',{className:'info'},info):'', err, el('div',{className:'btns'},cancel,save)));
  box._retour = retour || null;
  document.body.append(box);
  const first = box.querySelector('input,textarea,select'); if(first) first.focus();
}
function closeModal(){ const m=$('#modal'); if(!m) return; m.remove(); const back = m._retour; m._retour = null; if(back) back(); }
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });
const optionsOf = (values, current) => values.map(v=>el('option',{value:v[0],selected:v[0]===current},v[1]));
const LIENS_TYPES = [['conjoint','conjoint(e)'],['parent','parent'],['enfant','enfant'],['fratrie','frère ou sœur'],['demi','demi-frère ou demi-sœur'],['beauparent','beau-parent'],['beauenfant','beau-enfant'],['ami','ami(e)'],['collegue','collègue'],['voisin','voisin(e)'],['autre','autre (précision)']];
const TYPES = [['fait','fait'],['preference','préférence'],['tache','tâche'],['ephemere','éphémère'],['foyer','foyer (partagé)'],['relation','relation (partagée)']];
const peopleOptions = (current) => [['','— aucun (partagé) —'], ...PERSONS.filter(p=>!p.special).map(p=>[p.id,p.label])];

function editMemory(r){
  const txt = el('textarea',{value:r.texte});
  const typ = el('select',{}, ...optionsOf(TYPES, r.type));
  const own = el('select',{}, ...optionsOf(peopleOptions(), r.o));
  const tags = el('input',{value:r.tags.join(', ')});
  const imp = el('input',{type:'number',min:0,max:10,step:0.5,value:r.imp});
  const sv = el('input',{type:'number',min:0.1,max:2,step:0.05,value:r.s});
  const pin = el('input',{type:'checkbox',checked:r.tags.includes('pinned'),style:'width:auto'});
  const info = [`Identifiant ${r.id} · source ${r.src}`, `Créé ${r.cree} · dernier rappel ${r.dernier} · ${r.hits} rappel(s)`, `Importance effective ${r.eff} · état ${r.etat}`].join(' — ');
  openModal('Modifier le souvenir', [
    field('Texte', txt),
    el('div',{className:'row'}, field('Type', typ), field('Propriétaire', own)),
    field('Tags (séparés par des virgules)', tags),
    el('div',{className:'row'}, field('Note de départ (0 à 10)', imp), field('Renfort (0,1 à 2)', sv)),
    el('label',{}, pin, ' Protégé : jamais oublié (tag pinned)'),
  ], info, async ()=>{
    const list = tags.value.split(',').map(t=>t.trim()).filter(t=>t && t!=='pinned');
    if(pin.checked) list.push('pinned');
    const body = {id:r.id, texte:txt.value, type:typ.value, tags:list.join(','), imp:imp.value, s:sv.value};
    if(typ.value!=='foyer' && typ.value!=='relation') body.o = own.value;
    const x = await post('/api/update', body);
    return x.ok ? true : (x.erreur || 'Impossible (texte vide, mémoire verrouillée ?)');
  });
}
function editRelation(r){
  const rel = r.rel || {a:'',l:'',b:''};
  const a = el('input',{value:rel.a}), b = el('input',{value:rel.b});
  const type = rel.t || 'autre';
  const lt = el('select',{}, ...optionsOf(LIENS_TYPES, type));
  const prec = el('input',{value:rel.l, placeholder:'précision (obligatoire pour « autre »)'});
  const sv = el('input',{type:'number',min:0.1,max:2,step:0.05,value:r.s});
  openModal('Modifier la relation', [
    el('div',{className:'row'}, field('Personne', a), field('Lien (A est le … de B)', lt), field('Personne', b)),
    field('Précision (texte du lien, ex. femme de, marraine de)', prec),
    field('Renfort (0,1 à 2)', sv),
  ], `Identifiant ${r.id} · créée ${r.cree} · ${r.hits} rappel(s). Le texte du souvenir est recalculé : « A est le lien de B ».`, async ()=>{
    const x = await post('/api/update', {id:r.id, rel:{a:a.value,l:lt.value,b:b.value,precision:prec.value}, s:sv.value});
    return x.ok ? true : (x.erreur || 'Impossible');
  });
}
function editPerson(p, retour){
  const nom = el('input',{value:p.nom}), compl = el('input',{value:p.compl}), sur = el('input',{value:p.surnom});
  const sexe = el('select',{}, ...optionsOf([['','inconnu'],['f','féminin'],['m','masculin']], p.sexe||''));
  const connue = el('input',{type:'checkbox',checked:!p.citee});
  openModal('Modifier la personne', [
    el('div',{className:'row'}, field('Prénom', nom), field('Nom de famille', compl)),
    el('div',{className:'row'}, field('Surnom (comment elle veut être appelée)', sur), field('Sexe (pour dire fille, tante, grand-mère...)', sexe)),
    el('label',{className:'chk'}, connue, ' Connue par Aria (a déjà discuté avec elle : figure dans la liste des interlocuteurs)'),
  ], `Identifiant ${p.id} · créée ${p.cree} · ${p.souvenirs} souvenir(s)${p.actif?' · interlocuteur actif':''}${p.citee?' · seulement citée (pas encore interlocutrice)':''}`, async ()=>{
    const x = await post('/api/personne/update', {id:p.id, nom:nom.value, compl:compl.value, surnom:sur.value, sexe:sexe.value, connue:connue.checked});
    return x.ok ? true : (x.erreur || 'Le prénom est obligatoire');
  }, retour);
}
const SVGNS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs={}, ...kids) => { const e = document.createElementNS(SVGNS, tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]); kids.flat().forEach(c=>e.append(c)); return e; };
const CAT_COLOR = {famille:'var(--fam)', connaissance:'var(--con)'};
const CAT_NAME = {famille:'Famille', connaissance:'Connaissance'};

// Graphe des relations immédiates d'une personne : elle est au centre, ses proches autour ; un clic sur un proche le recentre
async function openGraph(pid, niv0){
  const box = el('div',{id:'graph-host'});
  const title = el('h2',{}, 'Relations');
  let courant = pid, niveau = niv0 || 1;
  const choix = el('select',{title:'Niveau des relations (1 = proches immédiats)'}, ...[1,2,3,4,5].map(n=>el('option',{value:String(n),selected:n===niveau}, 'Niveau ' + n)));
  choix.onchange = ()=>{ niveau = parseInt(choix.value, 10) || 1; show(courant); };
  const legend = el('div',{className:'legend'},
    el('span',{}, el('i',{style:'background:var(--fam)'}), 'Famille'),
    el('span',{}, el('i',{style:'background:var(--con)'}), 'Connaissance'),
    choix,
    el('span',{className:'mut'}, 'Cliquez sur une personne pour recentrer le graphe'));
  const close = el('button',{onclick:closeModal},'Fermer');
  closeModal();
  const modal = el('div',{id:'modal',onclick:e=>{if(e.target===modal)closeModal()}},
    el('div',{className:'card wide'}, title, legend, box, el('div',{className:'btns'}, close)));
  document.body.append(modal);

  async function show(id){
    courant = id;
    box.replaceChildren(el('p',{className:'mut'},'Chargement…'));
    const g = await get('/api/graphe?id=' + encodeURIComponent(id) + '&niveau=' + niveau);
    if(g.erreur){ box.replaceChildren(el('p',{className:'err'}, g.erreur)); return; }
    title.textContent = 'Relations de ' + g.centre.complet;
    drawGraph(box, g, show, ()=>editerCentre(g.centre.id));
  }
  // le clic sur la personne centrale ouvre son édition ; la fenêtre du graphe est rouverte (même niveau) à la fermeture
  function editerCentre(id){
    const p = PERSONS.find(x=>x.id===id);
    if(!p) return;
    modal.remove();
    editPerson(p, ()=>openGraph(id, niveau));
  }
  show(pid);
}

// Prénom, et nom de famille dessous s'il est connu (tronqués pour tenir dans le cercle)
function nodeText(node, x, y, nom, compl, centre){
  const cut = (t, n)=>t.length > n ? t.slice(0, n - 1) + '…' : t;
  const style = centre ? 'fill:var(--bg);font-weight:600' : '';
  const a = svgEl('text',{x:x,y:y + (compl ? -1 : 4),'text-anchor':'middle',style:style});
  a.textContent = cut(nom, 10);
  node.append(a);
  if(compl){
    const b = svgEl('text',{x:x,y:y + 12,'text-anchor':'middle',style:style + ';font-size:10px;opacity:.85'});
    b.textContent = cut(compl, 11);
    node.append(b);
  }
}

function drawGraph(box, g, onPick, onEditCentre){
  const niv = g.niveau || 1;
  const W = 720 + 140*(niv-1), H = 420 + 120*(niv-1), cx = W/2, cy = H/2;
  const svg = svgEl('svg',{id:'graph',viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'xMidYMid meet'});
  const step = Math.min((cx - 60), (cy - 46)) / niv;
  const rn = niv > 1 ? 28 : 31;
  // arbre : chaque personne est rattachée à celle d'où elle est atteinte ; son secteur d'angle est proportionnel à ses descendantes
  const enfants = {};
  g.voisins.forEach(v=>{ (enfants[v.de] = enfants[v.de] || []).push(v); });
  const poids = {};
  const peser = k=>{ poids[k] = (enfants[k]||[]).reduce((s,v)=>s + peser(v.n), 0) || 1; return poids[k]; };
  peser(g.centre.n);
  const pos = {}; pos[g.centre.n] = {x:cx, y:cy};
  const liste = [];
  const placer = (k, a0, a1)=>{
    let a = a0;
    (enfants[k]||[]).forEach(v=>{
      const part = (a1 - a0) * poids[v.n] / poids[k];
      const m = a + part/2, r = step * v.niveau;
      pos[v.n] = {x: cx + r*Math.cos(m), y: cy + r*Math.sin(m)};
      liste.push(v);
      placer(v.n, a, a + part);
      a += part;
    });
  };
  const depart = -Math.PI/2;
  placer(g.centre.n, depart, depart + 2*Math.PI);
  if(niv === 1 && liste.length === 1) pos[liste[0].n] = {x: cx + 200, y: cy};
  // arêtes d'abord (sous les nœuds), avec le libellé du lien au milieu
  liste.forEach(v=>{
    const p = pos[v.n], s = pos[v.de];
    const color = CAT_COLOR[v.categorie] || CAT_COLOR.connaissance;
    svg.append(svgEl('line',{x1:s.x,y1:s.y,x2:p.x,y2:p.y,stroke:color,'stroke-width':2.5,'stroke-linecap':'round',opacity:0.85}));
    const lab = svgEl('text',{x:(s.x+p.x)/2,y:(s.y+p.y)/2 - 4,'text-anchor':'middle',class:'edge-label'});
    lab.textContent = v.lien || '';
    svg.append(lab);
  });
  // proches
  liste.forEach(v=>{
    const p = pos[v.n];
    const color = CAT_COLOR[v.categorie] || CAT_COLOR.connaissance;
    const cliquable = !!v.id;                      // sans fiche (nom pas encore connu), on ne peut pas recentrer
    const node = svgEl('g',{class:'node',style:cliquable?'':'cursor:default'},
      svgEl('circle',{cx:p.x,cy:p.y,r:rn,fill:'var(--card)',stroke:color,'stroke-width':3}));
    nodeText(node, p.x, p.y, v.nom, v.compl, false);
    node.append(svgEl('title',{}, document.createTextNode(v.nom + ' — ' + (v.lien || 'lien') + ' (' + CAT_NAME[v.categorie] + ', niveau ' + v.niveau + ')' + (v.citee ? ' · jamais interlocuteur' : ''))));
    if(cliquable) node.addEventListener('click', ()=>onPick(v.id));
    svg.append(node);
  });
  // centre
  const ctr = svgEl('g',{class:'node'},
    svgEl('circle',{cx,cy,r:38,fill:'var(--ctr)',stroke:'var(--ctr)','stroke-width':3}));
  nodeText(ctr, cx, cy, g.centre.nom, g.centre.compl, true);
  if(onEditCentre){ ctr.style.cursor = 'pointer'; ctr.addEventListener('click', onEditCentre); ctr.append(svgEl('title',{}, document.createTextNode('Modifier la personne'))); }
  svg.append(ctr);
  const parts = [svg];
  if(!g.voisins.length) parts.unshift(el('p',{className:'mut'},'Aucune relation connue pour cette personne.'));
  box.replaceChildren(...parts);
}

function mergePerson(p){
  const others = PERSONS.filter(o=>!o.special && o.id!==p.id);
  if(!others.length){ say('Aucune autre personne vers qui fusionner'); return; }
  const target = el('select',{}, ...others.map(o=>el('option',{value:o.id},o.label)));
  openModal('Fusionner ' + p.label, [
    el('p',{}, 'Les ' + p.souvenirs + ' souvenir(s) de ' + p.label + ' et ses relations passent à la personne choisie ; ' + p.label + ' disparaît de la liste.'),
    field('Fusionner dans', target),
  ], 'Utile pour corriger un doublon (ex. deux fiches pour la même personne). Les souvenirs ne sont pas perdus.', async ()=>{
    const x = await post('/api/personne/merge', {id:p.id, vers:target.value});
    if(!x.ok) return 'Impossible';
    say(x.deplaces + ' souvenir(s) rattaché(s)'); return true;
  });
}

async function render(){
  const main=$('#main');main.replaceChildren();
  PERSONS=await get('/api/personnes');
  if(tab==='mem')return renderMem(main);
  if(tab==='per')return renderPersons(main);
  if(tab==='rel')return renderRelations(main);
  if(tab==='arc'){const rows=await get('/api/archive');
    return main.append(el('table',{},el('tr',{},...['Texte','Type','Raison','Date','Id',''].map(h=>el('th',{},h))),
      ...rows.map(r=>el('tr',{},el('td',{},r.txt),el('td',{},r.t),el('td',{},r.why),el('td',{className:'mut'},r.ts),el('td',{className:'mut'},r.id),
        el('td',{},el('button',{onclick:async()=>{say((await post('/api/restore',{id:r.id})).ok?'Restauré':'Impossible');render()}},'Restaurer'))))));}
  if(tab==='sta'){const s=await get('/api/stats');return main.append(el('pre',{},JSON.stringify(s,null,2)));}
  if(tab==='log'){const l=await get('/api/log');return main.append(el('pre',{},l.map(x=>JSON.stringify(x)).join('\\n')));}
}
async function renderMem(main){
  const rows=await get('/api/memories?q='+encodeURIComponent(query));
  const nbRel=rows.filter(r=>r.type==='relation').length;
  let list=rows.filter(r=>(fType?r.type===fType:r.type!=='relation')&&(!fState||r.etat===fState)&&(fOwner===''||(fOwner==='_foyer'?!r.o:r.o===fOwner)));
  list.sort((a,b)=>(a[sortKey]>b[sortKey]?1:-1)*sortDir);
  const inpQ=el('input',{placeholder:'Rechercher…',value:query,onchange:e=>{query=e.target.value;render()}});
  const selT=el('select',{onchange:e=>{fType=e.target.value;render()}},...['','fait','preference','tache','ephemere','foyer','relation'].map(v=>el('option',{value:v,selected:v===fType},v||'tous les types (hors relations)')));
  const selS=el('select',{onchange:e=>{fState=e.target.value;render()}},...['','actif','masque','protege'].map(v=>el('option',{value:v,selected:v===fState},v||'tous les états')));
  const selO=el('select',{onchange:e=>{fOwner=e.target.value;render()}},el('option',{value:''},'tous les propriétaires'),el('option',{value:'_foyer',selected:fOwner==='_foyer'},'Foyer (partagé)'),...PERSONS.filter(p=>!p.special).map(p=>el('option',{value:p.id,selected:p.id===fOwner},p.label)));
  const nowner=el('select',{},...PERSONS.filter(p=>!p.special).map(p=>el('option',{value:p.id},p.label)));
  const nt=el('input',{placeholder:'Nouveau souvenir…',size:40}),ntype=el('select',{},...['fait','preference','tache','ephemere','foyer'].map(v=>el('option',{value:v},v))),ntags=el('input',{placeholder:'tags',size:14});
  const add=el('button',{onclick:async()=>{if(!nt.value.trim())return;const r=await post('/api/add',{texte:nt.value,type:ntype.value,tags:ntags.value,o:nowner.value});say(r.statut);render()}},'Ajouter');
  const th=(k,t)=>el('th',{onclick:()=>{sortDir=(sortKey===k)?-sortDir:-1;sortKey=k;render()}},t+(sortKey===k?(sortDir>0?' ▲':' ▼'):''));
  const aide=(!fType&&nbRel)?el('p',{className:'mut'},nbRel+' relation(s) entre personnes ne sont pas listées ici : elles se gèrent dans l’onglet Relations. ',el('a',{href:'#',onclick:e=>{e.preventDefault();document.querySelector('nav button[data-t=rel]').click()}},'Ouvrir Relations')):'';
  main.append(el('div',{className:'bar'},inpQ,selT,selS,selO),aide,el('div',{className:'bar'},nt,ntype,ntags,el('span',{className:'mut'},'pour'),nowner,add),
    el('table',{},el('tr',{},th('texte','Souvenir'),th('type','Type'),th('proprietaire','Propriétaire'),th('eff','Importance'),th('tags','Tags'),th('cree','Créé'),th('hits','Rappels'),th('etat','État'),el('th',{},'')),
    ...list.map(r=>el('tr',{className:r.etat},
      el('td',{},r.texte,el('div',{className:'mut'},r.id+' · '+r.src)),el('td',{},r.type),
      el('td',{},ownerSelect(r)),
      el('td',{},String(r.eff),el('div',{className:'mut'},r.imp+' × '+r.s)),
      el('td',{},...r.tags.map(t=>el('span',{className:'tag'},t))),el('td',{className:'mut'},r.cree.slice(0,10)),el('td',{},String(r.hits)),el('td',{},r.etat),
      el('td',{},el('button',{title:'Modifier',onclick:()=>editMemory(r)},'✎'),' ',
        el('button',{title:'Protéger / déprotéger',onclick:async()=>{await post('/api/pin',{id:r.id});render()}},'📌'),' ',
        el('button',{title:'Oublier',onclick:async()=>{if(confirm('Oublier ce souvenir ?')){await post('/api/delete',{id:r.id});render()}}},'🗑'))))));
}
function ownerSelect(r){
  if(r.type==='relation'||r.type==='foyer')return el('span',{className:'mut'},'Foyer (partagé)');
  const sel=el('select',{onchange:async e=>{const x=await post('/api/update',{id:r.id,o:e.target.value});say(x.ok?'Attribué':(x.erreur||'Impossible'));render()}},
    el('option',{value:'',selected:!r.o},'— aucun (partagé) —'),...PERSONS.filter(p=>!p.special).map(p=>el('option',{value:p.id,selected:p.id===r.o},p.label)));
  return sel;
}
async function renderPersons(main){
  const nom=el('input',{placeholder:'Prénom',size:12}),compl=el('input',{placeholder:'Nom de famille',size:14}),sur=el('input',{placeholder:'Surnom',size:10});
  const add=el('button',{onclick:async()=>{if(!nom.value.trim())return;const r=await post('/api/personne/add',{nom:nom.value,compl:compl.value,surnom:sur.value});say(r.ok?'Ajoutée':'Impossible');render()}},'Ajouter');
  const people=PERSONS.filter(p=>!p.special);
  const inpP=el('input',{placeholder:'Rechercher une personne…',value:pq,onchange:e=>{pq=e.target.value;render()}});
  const barreP=el('div',{className:'bar'},inpP,
    listbox(pf,'champ',[['','Dans : tous les champs'],['nom','Dans : prénom'],['compl','Dans : nom de famille'],['surnom','Dans : surnom'],['id','Dans : identifiant']]),
    listbox(pf,'sexe',[['','Sexe : tous'],['f','Sexe : féminin'],['m','Sexe : masculin'],['?','Sexe : inconnu']]),
    listbox(pf,'statut',[['','Statut : tous'],['connue','Connues par Aria'],['citee','Seulement citées']]),
    listbox(pf,'souv',[['','Souvenirs : tous'],['avec','Avec souvenirs'],['sans','Sans souvenir']]),
    el('button',{title:'Effacer les filtres',onclick:()=>{pq='';pf={champ:'',sexe:'',statut:'',souv:''};render()}},'✕'));
  const filtreP=p=>match(pq,...(pf.champ?[p[pf.champ]]:[p.label,p.nom,p.compl,p.surnom,p.id]))
    &&(!pf.sexe||(pf.sexe==='?'?!p.sexe:p.sexe===pf.sexe))&&(!pf.statut||(pf.statut==='citee'?p.citee:!p.citee))
    &&(!pf.souv||(pf.souv==='avec'?p.souvenirs>0:p.souvenirs===0));
  const filtre=pq.trim()||pf.sexe||pf.statut||pf.souv||pf.champ;
  const liste=PERSONS.filter(p=>p.special?!filtre:filtreP(p));
  main.append(barreP,el('div',{className:'bar'},nom,compl,sur,add),
    el('table',{},el('tr',{},...['Personne','Prénom','Nom','Surnom','Sexe','Souvenirs',''].map(h=>el('th',{},h))),
      ...liste.map(p=>p.special?el('tr',{},el('td',{},p.label),el('td',{colSpan:4,className:'mut'},'souvenirs sans propriétaire (relations, foyer, anciens souvenirs)'),el('td',{},String(p.souvenirs)),el('td',{})):
        el('tr',{},el('td',{},p.label+(p.actif?' ● actif':'')+(p.citee?' (citée)':'')),el('td',{},p.nom),el('td',{},p.compl),el('td',{},p.surnom),el('td',{},p.sexe==='f'?'F':p.sexe==='m'?'M':'?'),el('td',{},String(p.souvenirs)),
          el('td',{},el('button',{title:'Graphe des relations',onclick:()=>openGraph(p.id)},'🕸'),' ',el('button',{title:'Modifier',onclick:()=>editPerson(p)},'✎'),' ',
            el('button',{title:'Fusionner dans une autre personne',onclick:()=>mergePerson(p)},'⇢'),' ',
            el('button',{title:'Supprimer',onclick:async()=>{if(!confirm('Supprimer '+p.label+' ?'))return;const r=await post('/api/personne/delete',{id:p.id});say(r.ok?'Supprimée':(r.erreur||'Impossible'));render()}},'🗑'))))));
}
async function renderRelations(main){
  const tous=await get('/api/relations');
  const gens=PERSONS.filter(p=>!p.special);
  const selA=el('select',{},el('option',{value:''},'Personne…'),...gens.map(p=>el('option',{value:p.id},p.label)));
  const selB=el('select',{},el('option',{value:''},'Personne…'),...gens.map(p=>el('option',{value:p.id},p.label)));
  const selL=el('select',{},...optionsOf(LIENS_TYPES,'conjoint'));
  const prec=el('input',{placeholder:'précision (pour « autre »)',size:20});
  const ajout=el('button',{onclick:async()=>{
    const r=await post('/api/relation/add',{a:selA.value,lien:selL.value,b:selB.value,precision:prec.value});
    say(r.ok?'Relation ajoutée':(r.erreur||'Impossible'));if(r.ok)render()}},'Ajouter');
  const creation=el('div',{className:'bar'},el('span',{className:'mut'},'Nouvelle relation :'),selA,el('span',{className:'mut'},'est'),selL,el('span',{className:'mut'},'de'),selB,prec,ajout);
  const quiOptions=[['','Personne : toutes'],...gens.map(p=>[p.id,'Personne : '+p.label])];
  const ab=r=>[(r.rel&&r.rel.a)||'',(r.rel&&r.rel.b)||''];
  const champsTxt=r=>({pers:ab(r),lien:[(r.rel&&r.rel.l)||''],texte:[r.texte]})[rf.champ]||[...ab(r),(r.rel&&r.rel.l)||'',r.texte,r.id];
  const rows=tous.filter(r=>match(rq,...champsTxt(r))
    &&(!rf.pers||(r.rel&&(r.rel.ia===rf.pers||r.rel.ib===rf.pers)))
    &&(!rf.lien||((r.rel&&r.rel.t)||'autre')===rf.lien)&&(!rf.cat||catDe(r)===rf.cat));
  const inpR=el('input',{placeholder:'Rechercher une relation…',value:rq,onchange:e=>{rq=e.target.value;render()}});
  const barreR=el('div',{className:'bar'},inpR,
    listbox(rf,'champ',[['','Dans : tous les champs'],['pers','Dans : personnes'],['lien','Dans : lien'],['texte','Dans : texte']]),
    listbox(rf,'pers',quiOptions),
    listbox(rf,'lien',[['','Lien : tous'],...LIENS_TYPES.map(l=>[l[0],'Lien : '+l[1]])]),
    listbox(rf,'cat',[['','Type : tous'],['famille','Famille'],['connaissance','Connaissance']]),
    el('button',{title:'Effacer les filtres',onclick:()=>{rq='';rf={champ:'',pers:'',lien:'',cat:''};render()}},'✕'),
    el('span',{className:'mut'},rows.length+' / '+tous.length));
  main.append(creation,barreR,el('p',{className:'mut'},'Liens entre personnes, partagés par tout le foyer et jamais oubliés. Aria les enregistre avec l\\'outil « relier ».'),
    el('table',{},el('tr',{},...['Personne','Lien','Personne','Souvenir',''].map(h=>el('th',{},h))),
      ...rows.map(r=>el('tr',{},el('td',{},(r.rel&&r.rel.a)||'?'),el('td',{},(r.rel&&r.rel.l)||''),el('td',{},(r.rel&&r.rel.b)||'?'),el('td',{className:'mut'},r.texte),
        el('td',{},el('button',{title:'Modifier',onclick:()=>editRelation(r)},'✎'),' ',el('button',{title:'Oublier',onclick:async()=>{if(confirm('Oublier cette relation ?')){await post('/api/delete',{id:r.id});render()}}},'🗑'))))));
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));render()});
render();
</script></body></html>
"""
