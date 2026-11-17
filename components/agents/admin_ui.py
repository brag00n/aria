"""Page d'administration des accès tiers : clés API, domaines autorisés, profils MCP. Serveur http.server (bibliothèque standard), port dédié
publié sur 127.0.0.1 seulement, intégré à l'interface d'Aria par un cadre (components/agents_tab.py).

Protections (reprises de tools/memory/ui.py, durcies) : en-tête Host dans une liste blanche (anti DNS rebinding), jeton `X-Token` généré
au démarrage et comparé à temps constant, `Origin` OBLIGATOIRE et identique à l'hôte sur tout POST, JSON seulement, corps ≤ 16 Ko,
textContent partout dans la page. La clé en clair n'est renvoyée qu'une fois, par la réponse de création, et n'est jamais journalisée.
"""
import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from components import audit_retention, echanges

MAX_BODY = 16 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


def make_handler(store, profiles, sessions, token, audit=None, extra_hosts=(), effacer_traces=None):
    allowed = {h.strip().lower() for h in extra_hosts if h.strip()}
    log = audit or (lambda *a, **k: None)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AccesUI"
        sys_version = ""

        def log_message(self, fmt, *args):
            pass

        def _host_ok(self):
            host = (self.headers.get("Host") or "").lower()
            name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
            return name in LOCAL_HOSTS or host in allowed or name in allowed

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
                pass

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False))

        def _guard(self, post, page=False):
            if not self._host_ok():
                self._json({"erreur": "hôte non autorisé"}, 403)
                return False
            if not page:
                sent = self.headers.get("X-Token") or ""
                if not hmac.compare_digest(sent.encode("utf-8"), token.encode("utf-8")):
                    self._json({"erreur": "jeton requis"}, 401)
                    return False
            if post:
                origin = self.headers.get("Origin")
                if not origin or urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
                    self._json({"erreur": "origine refusée"}, 403)          # l'origine est obligatoire : pas de repli
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

        # ---- lecture
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                if not self._guard(False, page=True):
                    return
                return self._send(200, PAGE.replace("__TOKEN__", token), "text/html; charset=utf-8")
            if not self._guard(False):
                return
            if url.path == "/api/keys":
                return self._json(store.list())
            if url.path == "/api/profils":
                out = []
                for name in profiles.names():
                    try:
                        out.append(profiles.describe(name))
                    except Exception as e:
                        out.append({"name": name, "erreur": str(e)[:120], "servers": []})
                return self._json(out)
            if url.path == "/api/sessions":
                return self._json(sessions.stats() if sessions else [])
            if url.path == "/api/journal":
                # Volumes et journal des purges seulement : cette page ne lit JAMAIS le contenu des conversations
                return self._json({"actif": echanges._settings["enabled"], "defaut_jours": echanges._settings["retention_days"],
                                   "texte": echanges._settings["log_text"], "volumes": audit_retention.volumes(),
                                   "purges": audit_retention.read_purges(50)[::-1],
                                   "acces": [{"id": k["id"], "label": k["label"], "retention_days": k.get("retention_days", 0)} for k in store.list()]})
            self._json({"erreur": "introuvable"}, 404)

        # ---- écriture
        def do_POST(self):
            if not self._guard(True):
                return
            body = self._body()
            if body is None:
                return self._json({"erreur": "corps JSON invalide"}, 400)
            path = urlparse(self.path).path
            kid = str(body.get("id", ""))
            if path == "/api/keys/create":
                profil = str(body.get("profile", ""))
                if profil and profil not in profiles.names():
                    return self._json({"ok": False, "erreur": "profil inconnu"})
                vue, cle = store.create(body.get("label", ""), body.get("origins") or [], profil, body.get("limits"),
                                        body.get("expires_days", 0), body.get("voice", True), body.get("retention_days", 0),
                                        body.get("publish_links", False), body.get("publish_speaker", False))
                if not vue:
                    return self._json({"ok": False, "erreur": "un nom est obligatoire"})
                log("key_created", id=vue["id"], profile=profil)
                return self._json({"ok": True, "key": vue, "secret": cle})       # la seule fois où la clé en clair est renvoyée
            if path == "/api/keys/update":
                fields = {k: body[k] for k in ("label", "origins", "profile", "limits", "voice", "retention_days", "publish_links", "publish_speaker", "expires_days") if k in body}
                if fields.get("profile") and fields["profile"] not in profiles.names():
                    return self._json({"ok": False, "erreur": "profil inconnu"})
                vue = store.update(kid, **fields)
                if vue and ("profile" in fields or "origins" in fields or "voice" in fields or "publish_links" in fields or "publish_speaker" in fields):
                    self._close_sessions(kid)
                log("key_updated", id=kid, fields=",".join(sorted(fields)))
                return self._json({"ok": bool(vue), "key": vue})
            if path == "/api/keys/revoke":
                ok = store.revoke(kid, bool(body.get("revoked", True)))
                if ok and body.get("revoked", True):
                    self._close_sessions(kid)
                log("key_revoked" if body.get("revoked", True) else "key_restored", id=kid)
                return self._json({"ok": ok})
            if path == "/api/keys/regenerate":
                vue, cle = store.regenerate(kid)
                if not vue:
                    return self._json({"ok": False, "erreur": "accès inconnu"})
                self._close_sessions(kid)          # les conversations ouvertes repartent proprement (la mémoire, elle, est conservée)
                log("key_regenerated", id=kid)
                return self._json({"ok": True, "key": vue, "secret": cle})        # la seule fois où la nouvelle clé en clair est renvoyée
            if path == "/api/keys/delete":
                ok = store.delete(kid)
                if ok:
                    self._close_sessions(kid)
                log("key_deleted", id=kid)
                return self._json({"ok": ok})
            if path == "/api/sessions/close":
                sid = str(body.get("id", ""))[:120]
                ok = bool(sessions and getattr(sessions, "request_close_session", None) and sessions.request_close_session(sid))
                log("session_closed", id=sid, ok=ok)
                return self._json({"ok": ok} if ok else {"ok": False, "erreur": "session inconnue ou déjà fermée"})
            if path == "/api/journal/purge-age":
                r = audit_retention.purge_auto(echanges._settings["retention_days"], store.retention_of, par="admin")
                log("audit_purge_age", lignes=r["lignes"], fichiers=r["fichiers"])
                return self._json({"ok": True, "lignes": r["lignes"], "fichiers": r["fichiers"]})
            if path == "/api/journal/purge":
                crit = {k: str(body.get(k) or "").strip()[:80] for k in ("date_avant", "conversation", "acces", "personne")}
                try:
                    r = audit_retention.purge_targeted(crit["date_avant"] or None, crit["conversation"] or None, crit["acces"] or None, crit["personne"] or None)
                except ValueError as e:
                    return self._json({"ok": False, "erreur": str(e)})
                traces = None
                if crit["personne"] and effacer_traces:
                    # Droit à l'effacement : la trace (extrait de la question) des souvenirs et fiches de cette personne est retirée aussi
                    try:
                        traces = effacer_traces(crit["personne"], r["ids"])
                    except Exception as e:
                        traces = {"erreur": type(e).__name__}
                    if isinstance(traces, dict) and "erreur" not in traces:
                        audit_retention.log_purge("admin", "traces de la personne", 0, 0, traces=traces.get("souvenirs", 0) + traces.get("personnes", 0))
                log("audit_purge", lignes=r["lignes"], fichiers=r["fichiers"])
                return self._json({"ok": True, "lignes": r["lignes"], "fichiers": r["fichiers"], "traces": traces})
            self._json({"erreur": "introuvable"}, 404)

        def _close_sessions(self, kid):
            """Les sessions ouvertes d'une clé révoquée, supprimée ou modifiée sont fermées (elles seront recréées avec le nouveau profil)."""
            if sessions and getattr(sessions, "request_close_key", None):
                sessions.request_close_key(kid)

    return Handler


def start(store, profiles, sessions, port, token=None, audit=None, extra_hosts=(), effacer_traces=None):
    """Lance la page dans un thread démon. Retourne (serveur, jeton)."""
    token = token or secrets.token_urlsafe(24)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store, profiles, sessions, token, audit, extra_hosts, effacer_traces))
    threading.Thread(target=httpd.serve_forever, name="acces-admin", daemon=True).start()
    return httpd, token


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Accès tiers</title>
<style>
:root{--bg:#fafafa;--fg:#1c1c1e;--mut:#6b6b70;--line:#d9d9de;--card:#fff;--acc:#2a6df4;--warn:#b45309;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececf0;--mut:#9a9aa3;--line:#33333a;--card:#1f1f23;--acc:#6c9bff;--warn:#f0a35a;--ok:#4ade80}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}nav button{margin-right:4px}
main{padding:12px 16px}
button,input,select,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{border-color:var(--acc);color:var(--acc)}
table{width:100%;border-collapse:collapse;background:var(--card)}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;white-space:nowrap}
.mut{color:var(--mut)}.tag{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 6px;margin:0 3px 3px 0;font-size:12px}
.bar{display:flex;gap:8px;margin:8px 0;flex-wrap:wrap;align-items:center}
.ok{color:var(--ok)}.ko{color:var(--warn)}
#msg{color:var(--warn);margin-left:8px}
code,.secret{font-family:ui-monospace,Consolas,monospace}
.secret{display:block;padding:8px;border:1px dashed var(--acc);border-radius:6px;word-break:break-all;margin:8px 0;user-select:all}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;z-index:10}
#modal .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;min-width:380px;max-width:92vw;max-height:90vh;overflow:auto}
#modal h2{font-size:15px;margin:0 0 10px}
#modal label{display:block;margin:8px 0 2px;color:var(--mut);font-size:12px}
#modal input,#modal select,#modal textarea{width:100%;box-sizing:border-box}
#modal textarea{min-height:70px;resize:vertical}
#modal .row{display:flex;gap:10px}#modal .row>div{flex:1}
#modal .info{margin-top:10px;font-size:12px;color:var(--mut)}
#modal .btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
#modal .err{color:var(--warn);margin-top:8px;min-height:1em}
#modal .chk{display:flex;align-items:center;gap:6px;margin-top:10px}#modal .chk input{width:auto}
</style></head><body>
<header><h1>🔑 Accès tiers</h1>
<nav><button data-t="keys" class="on">Clés</button><button data-t="prof">Profils MCP</button><button data-t="ses">Sessions</button><button data-t="jrn">Journal</button><button data-t="doc">Intégration</button></nav>
<span id="msg"></span></header>
<main id="main"></main>
<script>
const TOKEN='__TOKEN__';
const $=s=>document.querySelector(s);
const el=(tag,props={},...kids)=>{const e=document.createElement(tag);Object.assign(e,props);kids.flat().forEach(k=>e.append(k));return e};
const hdr=()=>({'Content-Type':'application/json','X-Token':TOKEN});
const get=async u=>(await fetch(u,{headers:hdr()})).json();
const post=async(u,b)=>(await fetch(u,{method:'POST',headers:hdr(),body:JSON.stringify(b)})).json();
const say=t=>{$('#msg').textContent=t;setTimeout(()=>{$('#msg').textContent=''},4000)};
let tab='keys',PROFILS=[];
const field=(label,node)=>el('div',{},el('label',{},label),node);
function closeModal(){const m=$('#modal');if(m)m.remove()}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});
function openModal(title,fields,info,onSave,saveLabel){
  closeModal();
  const err=el('div',{className:'err'});
  const save=el('button',{onclick:async()=>{err.textContent='';const r=await onSave();if(r===true){closeModal();render()}else if(r!=='keep')err.textContent=r||'Impossible'}},saveLabel||'Enregistrer');
  const box=el('div',{id:'modal',onclick:e=>{if(e.target===box)closeModal()}},
    el('div',{className:'card'},el('h2',{},title),...fields,info?el('div',{className:'info'},info):'',err,el('div',{className:'btns'},el('button',{onclick:closeModal},'Annuler'),save)));
  document.body.append(box);
  const first=box.querySelector('input,textarea,select');if(first)first.focus();
}
const lignes=t=>t.split(/[\\n,]+/).map(x=>x.trim()).filter(Boolean);
function profilSelect(courant){
  return el('select',{},el('option',{value:''},'(aucun : aucun outil)'),...PROFILS.map(p=>el('option',{value:p.name,selected:p.name===courant},p.name+(p.description?' - '+p.description:''))));
}
// Copie dans le presse-papiers. navigator.clipboard n'existe que sur une page sécurisée (https ou localhost) : cette page est servie en http sur
// 127.0.0.1 dans un cadre, donc repli par sélection du texte et document.execCommand. Si rien ne marche, le texte reste sélectionné pour Ctrl+C.
async function copier(texte,zone){
  const selectionner=()=>{const r=document.createRange();r.selectNodeContents(zone);const s=getSelection();s.removeAllRanges();s.addRange(r)};
  try{ if(navigator.clipboard&&window.isSecureContext){ await navigator.clipboard.writeText(texte); say('Copiée dans le presse-papiers'); return } }catch(e){}
  selectionner();
  let ok=false;
  try{ ok=document.execCommand('copy') }catch(e){}
  say(ok?'Copiée dans le presse-papiers':'Copie impossible : la clé est sélectionnée, faites Ctrl+C');
}
function secretModal(titre,secret){
  closeModal();
  const zone=el('code',{className:'secret'},secret);
  const box=el('div',{id:'modal'},el('div',{className:'card'},el('h2',{},titre),
    el('p',{},'Copiez cette clé maintenant : elle ne sera plus jamais affichée.'),
    zone,
    el('div',{className:'btns'},el('button',{onclick:()=>copier(secret,zone)},'Copier'),
      el('button',{onclick:()=>{closeModal();render()}},'J\\u2019ai copié la clé'))));
  document.body.append(box);
}
function editKey(k){
  const neuf=!k;
  const label=el('input',{value:k?k.label:'',placeholder:'Nom de l\\u2019application'});
  const origins=el('textarea',{value:k?k.origins.join('\\n'):'',placeholder:'https://app.example.com (une par ligne)'});
  const prof=profilSelect(k?k.profile:'');
  const rate=el('input',{type:'number',min:1,value:k?k.limits.rate_per_min:30});
  const day=el('input',{type:'number',min:1,value:k?k.limits.daily_turns:500});
  const exp=el('input',{type:'number',min:0,value:'0',title:'0 = sans expiration'});
  const voice=el('input',{type:'checkbox',checked:k?k.voice:true});
  const publish=el('input',{type:'checkbox',checked:k?!!k.publish_links:false});
  const speaker=el('input',{type:'checkbox',checked:k?!!k.publish_speaker:false});
  const reten=el('input',{type:'number',min:0,max:3650,value:k?(k.retention_days||0):0,title:'0 = durée par défaut du journal'});
  openModal(neuf?'Nouvelle clé':'Modifier la clé',[
    field('Nom',label),field('Domaines autorisés (une origine exacte par ligne)',origins),field('Profil MCP (outils de cette application)',prof),
    el('div',{className:'row'},field('Requêtes par minute',rate),field('Tours par jour',day)),
    field(neuf?'Expire dans (jours, 0 = jamais)':'Prolonger : expire dans (jours, 0 = jamais)',exp),
    el('label',{className:'chk'},voice,' Voix autorisée (micro et synthèse vocale)'),
    el('label',{className:'chk'},publish,' Peut publier ses propres liens : « moi est l\u2019ami de Marion » devient visible de tous (les fiches nouvelles sont à valider ; jamais de lien entre d\u2019autres personnes)'),
    el('label',{className:'chk'},speaker,' Peut publier les informations de son interlocuteur : l\u2019accès a les droits d\u2019Aria pour la personne qui lui parle (souvenirs à son nom, liens entre n\u2019importe quelles personnes). Cette personne n\u2019est PAS authentifiée : réservé aux applications de confiance'),
    field('Conservation des échanges dans le journal d\u2019audit (jours, 0 = durée par défaut)',reten),
  ],k?('Identifiant '+k.id+' \\u00b7 préfixe '+k.prefix+' \\u00b7 '+k.etat):'La clé ne sera affichée qu\\u2019une seule fois, à la création.',async()=>{
    const corps={label:label.value,origins:lignes(origins.value),profile:prof.value,limits:{rate_per_min:+rate.value,daily_turns:+day.value},voice:voice.checked,publish_links:publish.checked,publish_speaker:speaker.checked,retention_days:+reten.value||0};
    if(neuf){corps.expires_days=+exp.value;const r=await post('/api/keys/create',corps);if(!r.ok)return r.erreur;closeModal();secretModal('Clé créée',r.secret);return 'keep'}
    corps.id=k.id;if(+exp.value>0||exp.value==='0'&&k.expires)corps.expires_days=+exp.value;
    const r=await post('/api/keys/update',corps);return r.ok?true:'Impossible';
  },neuf?'Créer':'Enregistrer');
}
async function renderKeys(main){
  const rows=await get('/api/keys');
  main.append(el('div',{className:'bar'},el('button',{onclick:()=>editKey(null)},'+ Nouvelle clé'),
    el('span',{className:'mut'},'Les applications tierces appellent l\\u2019API avec cette clé (Authorization: Bearer ak_…).')),
    el('table',{},el('tr',{},...['Nom','Clé','État','Profil','Domaines','Limites','Dernier usage',''].map(h=>el('th',{},h))),
      ...rows.map(k=>el('tr',{},el('td',{},k.label,el('div',{className:'mut'},k.id)),el('td',{},el('code',{},k.prefix+'…')),
        el('td',{className:k.etat==='active'?'ok':'ko'},k.etat+(k.expires?' (expire '+k.expires.slice(0,10)+')':'')),
        el('td',{},k.profile||el('span',{className:'ko'},'aucun')),
        el('td',{},...(k.origins.length?k.origins.map(o=>el('span',{className:'tag'},o)):[el('span',{className:'ko'},'aucun navigateur')])),
        el('td',{className:'mut'},k.limits.rate_per_min+'/min · '+k.limits.daily_turns+'/jour'+(k.voice?' · voix':' · texte')+(k.publish_links?' · publie ses liens':'')+(k.publish_speaker?' · agit pour son interlocuteur':'')),
        el('td',{className:'mut'},k.last_used?k.last_used.replace('T',' ').slice(0,16):'jamais'),
        el('td',{},el('button',{title:'Modifier',onclick:()=>editKey(k)},'✎'),' ',
          el('button',{title:'Régénérer la clé API (la mémoire et les réglages de l\\u2019accès sont conservés ; l\\u2019ancienne clé cesse de fonctionner)',onclick:async()=>{if(confirm('Régénérer la clé de « '+k.label+' » ? L\\u2019ancienne clé cessera de fonctionner immédiatement. La mémoire de l\\u2019accès est conservée.')){const r=await post('/api/keys/regenerate',{id:k.id});if(r.ok)secretModal('Nouvelle clé de « '+k.label+' »',r.secret);else say(r.erreur||'Impossible')}}},'🔄'),' ',
          el('button',{title:k.revoked?'Réactiver':'Révoquer',onclick:async()=>{await post('/api/keys/revoke',{id:k.id,revoked:!k.revoked});render()}},k.revoked?'↺':'⛔'),' ',
          el('button',{title:'Supprimer',onclick:async()=>{if(confirm('Supprimer définitivement la clé « '+k.label+' » ?')){await post('/api/keys/delete',{id:k.id});render()}}},'🗑'))))));
}
async function renderProfils(main){
  main.append(el('p',{className:'mut'},'Un profil choisit les serveurs MCP, le message système et le LLM d\\u2019une application. Il se gère dans l\\u2019onglet Paramètres d\\u2019Aria (section « Accès tiers : profils ») et accède à la mémoire d\\u2019Aria sous l\\u2019identité de son accès (une personne système aux souvenirs privés).'),
    el('table',{},el('tr',{},...['Profil','Description','LLM','Serveurs MCP','Voix'].map(h=>el('th',{},h))),
      ...PROFILS.map(p=>el('tr',{},el('td',{},p.name),el('td',{},p.erreur?el('span',{className:'ko'},p.erreur):p.description||''),el('td',{className:'mut'},p.llm||'(module actif)'),
        el('td',{},...p.servers.map(s=>el('span',{className:'tag'},s.name+' · '+s.type))),el('td',{},p.voice===false?'non':'oui')))));
  if(!PROFILS.length)main.append(el('p',{className:'ko'},'Aucun profil. Créez-en un dans l\\u2019onglet Paramètres, section « Accès tiers : profils ».'));
}
async function renderSessions(main){
  const rows=await get('/api/sessions');
  main.append(el('table',{},el('tr',{},...['Session','Clé','Profil','Interlocuteur','Inactive depuis',''].map(h=>el('th',{},h))),
    ...rows.map(s=>el('tr',{},el('td',{},s.id),el('td',{},s.key),el('td',{},s.profile),
      el('td',{className:s.speaker?'':'mut'},s.speaker===null?'–':(s.speaker||'inconnu')),el('td',{className:'mut'},s.idle_s+' s'),
      el('td',{},el('button',{title:'Fermer cette session et libérer sa place (le journal d\u2019audit et les souvenirs sont conservés)',onclick:async()=>{if(!confirm('Fermer la session « '+s.id+' » ?'))return;const r=await post('/api/sessions/close',{id:s.id});say(r.ok?'Session fermée':(r.erreur||'Impossible'));setTimeout(render,400)}},'🗑'))))));
  if(!rows.length)main.append(el('p',{className:'mut'},'Aucune session ouverte.'));
}
async function renderJournal(main){
  const j=await get('/api/journal');
  const nb=j.volumes.reduce((a,v)=>a+v.lignes,0);
  const date=el('input',{type:'date'}),conv=el('input',{placeholder:'identifiant exact (ex. k_2b7cfa:abc#1)'}),pers=el('input',{placeholder:'nom exact de l\u2019interlocuteur'});
  const acc=el('select',{},el('option',{value:''},'(tous)'),el('option',{value:'aria'},'Aria (interface)'),...j.acces.map(a=>el('option',{value:a.id},a.label+' \u00b7 '+a.id)));
  const lancer=async()=>{
    const corps={date_avant:date.value,conversation:conv.value,acces:acc.value,personne:pers.value};
    if(!corps.date_avant&&!corps.conversation&&!corps.acces&&!corps.personne){say('Choisissez au moins un critère');return}
    if(!confirm('Supprimer définitivement les échanges qui correspondent à ces critères'+(corps.personne?' et effacer les traces de cette personne dans la mémoire':'')+' ?'))return;
    const r=await post('/api/journal/purge',corps);
    say(r.ok?(r.lignes+' échange(s) supprimé(s)'+(r.traces&&r.traces.erreur?' \u00b7 traces : erreur':'')):(r.erreur||'Impossible'));render();
  };
  main.append(
    el('p',{className:'mut'},(j.actif?'Journal actif':'Journal désactivé')+' \u00b7 conservation par défaut '+j.defaut_jours+' jours (réglable par accès) \u00b7 '+(j.texte?'texte complet':'empreintes seulement')+' \u00b7 '+nb+' échange(s). Cette page n\u2019affiche jamais le contenu des conversations.'),
    el('div',{className:'bar'},el('button',{onclick:async()=>{const r=await post('/api/journal/purge-age',{});say(r.lignes+' échange(s) purgé(s)');render()}},'Purger maintenant selon la durée de conservation')),
    el('h3',{},'Purge ciblée'),
    el('div',{className:'bar'},el('label',{},'Avant le ',date),el('label',{},'Conversation ',conv),el('label',{},'Accès ',acc),el('label',{},'Personne ',pers),el('button',{onclick:lancer},'Supprimer')),
    el('h3',{},'Volumes par jour'),
    el('table',{},el('tr',{},...['Jour','Échanges','Taille'].map(h=>el('th',{},h))),...j.volumes.slice().reverse().map(v=>el('tr',{},el('td',{},v.jour),el('td',{},String(v.lignes)),el('td',{className:'mut'},Math.round(v.octets/1024)+' Ko')))),
    j.volumes.length?'':el('p',{className:'mut'},'Aucun échange consigné.'),
    el('h3',{},'Journal des purges'),
    el('table',{},el('tr',{},...['Date','Par','Critère','Échanges','Fichiers','Période','Traces'].map(h=>el('th',{},h))),
      ...j.purges.map(p=>el('tr',{},el('td',{},p.ts.replace('T',' ').slice(0,16)),el('td',{},p.par),el('td',{},p.critere),el('td',{},String(p.lignes)),el('td',{},String(p.fichiers)),
        el('td',{className:'mut'},p.du?p.du.slice(0,10)+' \u2192 '+p.au.slice(0,10):''),el('td',{},String(p.traces||0))))),
    j.purges.length?'':el('p',{className:'mut'},'Aucune purge.'));
}
function renderDoc(main){
  const base=location.protocol+'//'+location.hostname+':7861';
  main.append(el('p',{},'Intégration dans une page web tierce :'),
    el('pre',{},'<scr'+'ipt src="'+base+'/aria-client.js"><'+'/script>\\n<scr'+'ipt>\\n  const aria = AriaClient.create({baseUrl: "'+base+'", apiKey: "ak_…", voice: true});\\n  aria.on("text", t => console.log(t));\\n  await aria.start();        // écoute micro\\n  aria.say("Bonjour Aria");  // saisie texte\\n<'+'/script>'),
    el('p',{className:'mut'},'Le navigateur de la page tierce doit avoir accepté une fois le certificat de '+base+'. Le micro exige une page en HTTPS (ou localhost).'));
}
async function render(){
  const main=$('#main');main.replaceChildren();
  PROFILS=await get('/api/profils');
  if(tab==='keys')return renderKeys(main);
  if(tab==='prof')return renderProfils(main);
  if(tab==='ses')return renderSessions(main);
  if(tab==='jrn')return renderJournal(main);
  return renderDoc(main);
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));render()});
render();
</script></body></html>
"""
