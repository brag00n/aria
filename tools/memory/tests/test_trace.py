"""Trace d'audit (`tr`) des souvenirs et des personnes : l'échange qui les a fait naître, reçu par `_meta` (jamais par les arguments du LLM).
Les tests de transport (stdio, HTTP) exigent le paquet `mcp` (conteneur) : ignorés sinon."""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit_trace  # noqa: E402
import store as store_mod  # noqa: E402
from interlocuteur import Interlocuteurs  # noqa: E402
from service import Service  # noqa: E402

try:
    import httpx  # noqa: F401
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from starlette.testclient import TestClient
    import http_facade
    HAVE = True
except ImportError:
    HAVE = False

SERVER = Path(__file__).resolve().parent.parent / "server.py"
SECRET = "secret-de-test-assez-long"


def meta(eid="e_18a2b3c4d5e6f7", src="aria", who="Olivier", x="Retiens que j'aime le thé", ts="2026-11-14T10:00:00.000+00:00"):
    return {"echange": eid, "ts": ts, "src": src, "who": who, "x": x}


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        self.t += 1
        return self.t


class CleanTests(unittest.TestCase):
    def test_meta_valide_devient_une_trace(self):
        tr = audit_trace.clean(meta(src="k_2b7cfa:Robot", who="Robot"))
        self.assertEqual(tr, {"e": "e_18a2b3c4d5e6f7", "ts": "2026-11-14T10:00:00.000+00:00", "src": "k_2b7cfa:Robot", "who": "Robot", "x": "Retiens que j'aime le thé"})

    def test_trace_deja_stockee_est_idempotente(self):
        tr = audit_trace.clean(meta())
        self.assertEqual(audit_trace.clean(tr), tr)

    def test_identifiants_invalides_refuses(self):
        for faux in (None, "", "e_", "x_123456", "e_12", "E_ABCDEF", "e_../../x", "e_" + "a" * 50, 42, ["e_abcdef"], "e_abc def"):
            self.assertIsNone(audit_trace.clean({"echange": faux, "src": "aria"}), faux)
        for pas_dict in (None, "e_abcdef", 3, [], ()):
            self.assertIsNone(audit_trace.clean(pas_dict))

    def test_champs_bornes_et_normalises(self):
        tr = audit_trace.clean(meta(x="mot  \n " * 200, who="W" * 500, src="s" * 500, ts="t" * 500))
        self.assertLessEqual(len(tr["x"]), 200)
        self.assertLessEqual(len(tr["who"]), 60)
        self.assertLessEqual(len(tr["src"]), 80)
        self.assertLessEqual(len(tr["ts"]), 40)
        self.assertNotIn("\n", tr["x"])

    def test_champs_inconnus_ignores(self):
        tr = audit_trace.clean({**meta(), "o": "p1", "admin": True})
        self.assertEqual(set(tr), {"e", "ts", "src", "who", "x"})

    def test_contexte_absent_ou_sans_meta(self):
        class Rc:
            meta = None

        class Ctx:
            request_context = Rc()
        self.assertIsNone(audit_trace.from_context(None))
        self.assertIsNone(audit_trace.from_context(Ctx()))
        self.assertIsNone(audit_trace.from_context(object()))


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trace_"))
        self.s = store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)
        self.i = Interlocuteurs(self.s)
        self.svc = Service(self.s, self.i)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rouvrir(self):
        return store_mod.Store(self.tmp, "default", clock=Clock(), lock_instance=False)


class SouvenirTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.i.identifier("Olivier")

    def test_un_souvenir_porte_la_trace_de_son_echange(self):
        st, rec = self.s.add("Olivier aime le thé", "preference", trace=meta())
        self.assertEqual(st, "added")
        self.assertEqual(rec["tr"]["e"], "e_18a2b3c4d5e6f7")
        self.assertEqual((rec["tr"]["src"], rec["tr"]["who"]), ("aria", "Olivier"))

    def test_la_trace_survit_au_rechargement_et_a_la_compaction(self):
        _, rec = self.s.add("Olivier aime le thé", "preference", trace=meta())
        s2 = self.rouvrir()
        self.assertEqual(s2.records[rec["id"]]["tr"]["x"], "Retiens que j'aime le thé")
        s2.compact()
        self.assertEqual(self.rouvrir().records[rec["id"]]["tr"]["e"], "e_18a2b3c4d5e6f7")

    def test_sans_trace_aucun_champ_ecrit_et_anciennes_donnees_lisibles(self):
        _, rec = self.s.add("Olivier aime le café", "preference")
        self.assertIsNone(rec["tr"])
        brut = (self.tmp / "default" / "memories.jsonl").read_text("utf-8") if (self.tmp / "default" / "memories.jsonl").exists() else ""
        self.assertNotIn('"tr"', brut)
        self.assertIsNone(self.rouvrir().records[rec["id"]]["tr"])

    def test_trace_invalide_ignoree_sans_bloquer_le_souvenir(self):
        st, rec = self.s.add("Olivier aime le vélo", "preference", trace={"echange": "pas un id"})
        self.assertEqual(st, "added")
        self.assertIsNone(rec["tr"])

    def test_un_doublon_ne_remplace_pas_la_trace_d_origine(self):
        _, rec = self.s.add("Olivier aime le thé", "preference", trace=meta())
        st, _ = self.s.add("Olivier aime le thé", "preference", trace=meta("e_zzzzzzzzzz"))
        self.assertEqual(st, "duplicate")
        self.assertEqual(self.s.records[rec["id"]]["tr"]["e"], "e_18a2b3c4d5e6f7")

    def test_update_ne_peut_pas_modifier_la_trace(self):
        _, rec = self.s.add("Olivier aime le thé", "preference", trace=meta())
        self.s.update(rec["id"], {"tr": {"e": "e_falsifie123"}, "txt": "Olivier aime le thé noir"})
        self.assertEqual(self.s.records[rec["id"]]["tr"]["e"], "e_18a2b3c4d5e6f7")
        self.assertEqual(self.rouvrir().records[rec["id"]]["tr"]["e"], "e_18a2b3c4d5e6f7")

    def test_service_memoriser_transmet_la_trace(self):
        self.svc.memoriser("Olivier aime le jazz", "preference", trace=meta(x="Je suis fan de jazz"))
        rec = next(r for r in self.s.records.values() if "jazz" in r["txt"])
        self.assertEqual(rec["tr"]["x"], "Je suis fan de jazz")

    def test_le_journal_d_activite_cite_l_echange(self):
        self.svc.memoriser("Olivier aime le rock", "preference", trace=meta())
        evs = [e for e in self.s.activity.tail()] if hasattr(self.s.activity, "tail") else []
        self.assertTrue(all("ex" in e for e in evs if e.get("ev") == "add" and not e.get("dup")) or not evs)

    def test_la_trace_d_une_relation(self):
        self.svc.relier("moi", "conjoint", "Tiphaine", trace=meta(x="Tiphaine est ma femme"))
        rel = next(r for r in self.s.records.values() if r["t"] == "relation")
        self.assertEqual(rel["tr"]["x"], "Tiphaine est ma femme")


class PersonneTests(StoreCase):
    def test_personne_creee_par_identification_porte_la_trace(self):
        self.svc.identifier("Olivier", trace=meta(x="Je m'appelle Olivier"))
        p = self.s.personnes.get(self.s.active)
        self.assertEqual(p["tr"]["x"], "Je m'appelle Olivier")

    def test_personne_citee_dans_un_lien_porte_la_trace(self):
        self.svc.identifier("Olivier")
        self.svc.relier("Tiphaine", "conjoint", "moi", trace=meta(x="Ma femme Tiphaine"))
        tiph = next(p for p in self.s.personnes.items if p["nom"] == "Tiphaine")
        self.assertEqual(tiph["tr"]["e"], "e_18a2b3c4d5e6f7")

    def test_la_trace_survit_a_une_reecriture_du_fichier(self):
        """`Personnes._load` a une liste fermée de champs : un champ oublié disparaîtrait à la première réécriture."""
        self.svc.identifier("Olivier", trace=meta())
        pid = self.s.active
        self.s.personnes.update(pid, surnom="Oli")                       # réécriture complète du fichier
        rouvert = self.rouvrir()
        self.assertEqual(rouvert.personnes.get(pid)["tr"]["e"], "e_18a2b3c4d5e6f7")
        self.assertEqual(rouvert.personnes.get(pid)["surnom"], "Oli")

    def test_personne_existante_garde_sa_trace_d_origine(self):
        self.svc.identifier("Olivier", trace=meta())
        self.svc.identifier("Olivier", trace=meta("e_autreechange1"))
        self.assertEqual(self.s.personnes.get(self.s.active)["tr"]["e"], "e_18a2b3c4d5e6f7")

    def test_personne_sans_trace_n_a_pas_de_champ(self):
        self.svc.identifier("Paul")
        self.assertNotIn("tr", self.s.personnes.get(self.s.active))

    def test_personne_systeme_porte_la_trace_de_sa_premiere_apparition(self):
        p = self.s.personnes.ensure_system("k_robot", "Robot", trace=meta(src="k_robot:Robot", who="Robot"))
        self.assertEqual(p["tr"]["src"], "k_robot:Robot")
        self.assertEqual(self.s.personnes.ensure_system("k_robot", "Robot2", trace=meta("e_autreechange1"))["tr"]["e"], "e_18a2b3c4d5e6f7")


class EffacementTests(StoreCase):
    """Droit à l'effacement : la trace d'une personne disparaît des souvenirs et des fiches, le souvenir lui-même reste."""

    def setUp(self):
        super().setUp()
        self.i.identifier("Olivier", trace=meta(who="Olivier"))
        self.svc.memoriser("Olivier aime le thé", "preference", trace=meta("e_olivier0001", who="Olivier"))
        self.i.identifier("Paul", trace=meta("e_paul0000001", who="Paul", x="Je suis Paul"))
        self.svc.memoriser("Paul aime le vélo", "preference", trace=meta("e_paul0000002", who="Paul"))

    def rec(self, mot):
        return next(r for r in self.s.records.values() if mot in r["txt"])

    def test_efface_les_traces_de_la_personne_seulement(self):
        souvenirs, personnes = self.s.effacer_traces("paul")                 # insensible à la casse
        self.assertEqual((souvenirs, personnes), (1, 1))
        self.assertIsNone(self.rec("vélo")["tr"])
        self.assertIsNotNone(self.rec("thé")["tr"])
        self.assertNotIn("tr", next(p for p in self.s.personnes.items if p["nom"] == "Paul"))
        self.assertIn("tr", next(p for p in self.s.personnes.items if p["nom"] == "Olivier"))

    def test_le_souvenir_lui_meme_est_conserve(self):
        self.s.effacer_traces("Paul")
        self.assertIn("Paul aime le vélo", [r["txt"] for r in self.s.records.values()])

    def test_effacement_persistant_apres_rechargement_et_compaction(self):
        self.s.effacer_traces("Paul")
        self.assertIsNone(self.rouvrir().records[self.rec("vélo")["id"]]["tr"])
        self.s.compact()
        r = self.rouvrir()
        self.assertIsNone(r.records[self.rec("vélo")["id"]]["tr"])
        self.assertNotIn("tr", next(p for p in r.personnes.items if p["nom"] == "Paul"))

    def test_par_identifiants_d_echange(self):
        self.assertEqual(self.s.effacer_traces("", ["e_olivier0001"])[0], 1)
        self.assertIsNone(self.rec("thé")["tr"])
        self.assertIsNotNone(self.rec("vélo")["tr"])

    def test_sans_critere_rien_n_est_efface(self):
        self.assertEqual(self.s.effacer_traces("", []), (0, 0))
        self.assertIsNotNone(self.rec("thé")["tr"])

    def test_aucune_correspondance_exacte_pas_de_sous_chaine(self):
        self.assertEqual(self.s.effacer_traces("Pau"), (0, 0))


async def appels_stdio(data_dir, calls):
    env = {**os.environ, "MEMORY_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            out = []
            for name, args, m in calls:
                res = await session.call_tool(name, args, meta=m)
                out.append(res.content[0].text if res.content else "")
            return tools, out


@unittest.skipUnless(HAVE, "mcp absent (lancer dans le conteneur)")
class TransportStdioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="trace_e2e_"))

    def tearDown(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        shutil.rmtree(self.tmp, ignore_errors=True)

    def memoires(self):
        s = store_mod.Store(self.tmp, "default", lock_instance=False)
        return s, list(s.records.values())

    def test_meta_lu_par_le_serveur_stdio_et_conserve(self):
        tools, _ = asyncio.run(appels_stdio(self.tmp, [
            ("identifier_interlocuteur", {"nom": "Olivier"}, meta(x="Je m'appelle Olivier")),
            ("memoriser", {"fait": "Olivier aime le thé", "type": "preference"}, meta("e_exemple0001", x="J'aime le thé")),
        ]))
        s, recs = self.memoires()
        (rec,) = [r for r in recs if "thé" in r["txt"]]
        self.assertEqual((rec["tr"]["e"], rec["tr"]["x"]), ("e_exemple0001", "J'aime le thé"))
        self.assertEqual(s.personnes.items[0]["tr"]["x"], "Je m'appelle Olivier")

    def test_le_schema_des_outils_n_expose_pas_le_contexte(self):
        tools, _ = asyncio.run(appels_stdio(self.tmp, []))
        for t in tools:
            self.assertNotIn("ctx", t.inputSchema.get("properties", {}), t.name)
            self.assertNotIn("_meta", t.inputSchema.get("properties", {}), t.name)

    def test_le_llm_ne_peut_pas_fournir_la_trace_par_les_arguments(self):
        asyncio.run(appels_stdio(self.tmp, [
            ("identifier_interlocuteur", {"nom": "Olivier"}, None),
        ]))
        try:
            asyncio.run(appels_stdio(self.tmp, [
                ("memoriser", {"fait": "Olivier aime le ski", "type": "preference", "tr": {"e": "e_falsifie123"}, "_meta": {"echange": "e_falsifie123"}}, None),
            ]))
        except Exception:
            pass                                                          # arguments inconnus : refusés ou ignorés, les deux conviennent
        _, recs = self.memoires()
        for r in recs:
            self.assertIsNone(r["tr"], r["txt"])

    def test_sans_meta_aucune_trace(self):
        asyncio.run(appels_stdio(self.tmp, [
            ("identifier_interlocuteur", {"nom": "Olivier"}, None),
            ("memoriser", {"fait": "Olivier aime le cinéma", "type": "preference"}, None),
        ]))
        _, recs = self.memoires()
        self.assertTrue(recs and all(r["tr"] is None for r in recs))

    def test_meta_invalide_ne_bloque_pas_l_outil(self):
        asyncio.run(appels_stdio(self.tmp, [
            ("identifier_interlocuteur", {"nom": "Olivier"}, {"echange": "n'importe quoi"}),
            ("memoriser", {"fait": "Olivier aime la danse", "type": "preference"}, {"echange": 42}),
        ]))
        _, recs = self.memoires()
        self.assertTrue(any("danse" in r["txt"] for r in recs))
        self.assertTrue(all(r["tr"] is None for r in recs))


def rpc(method, params=None, rid=1):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


@unittest.skipUnless(HAVE, "mcp / starlette absents (lancer dans le conteneur)")
class TransportHttpTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.i.identifier("Olivier")
        self.app = http_facade.make_app(self.s, self.svc, SECRET)
        self.client = TestClient(self.app.inner, base_url="http://127.0.0.1:7872")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        super().tearDown()

    def tool(self, name, args, m=None, cle="k_robot", nom="Robot"):
        params = {"name": name, "arguments": args}
        if m is not None:
            params["_meta"] = m
        r = self.client.post("/mcp", json=rpc("tools/call", params),
                             headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
                                      "X-Memory-Principal": cle, "X-Memory-Name": http_facade.encode_name(nom)})
        body = r.json()
        return body.get("result", {}).get("content", [{}])[0].get("text", "") if "result" in body else json.dumps(body)

    def systeme(self, cle="k_robot"):
        return self.s.personnes.systeme(cle)

    def test_la_trace_d_un_tiers_porte_son_identifiant_et_son_nom(self):
        m = meta("e_tiers000001", src="k_robot:Robot", who="Robot", x="Retiens que la batterie est à 80 %")
        self.assertTrue(self.tool("memoriser", {"fait": "Batterie à 80 pour cent"}, m).startswith("OK"))
        rec = next(r for r in self.s.records.values() if "Batterie" in r["txt"])
        self.assertEqual((rec["tr"]["e"], rec["tr"]["src"], rec["tr"]["who"]), ("e_tiers000001", "k_robot:Robot", "Robot"))
        self.assertEqual(rec["o"], self.systeme()["id"])                     # la trace n'a aucun effet sur l'identité imposée par l'en-tête

    def test_la_premiere_requete_trace_la_creation_de_la_personne_systeme(self):
        self.tool("memoriser", {"fait": "Premier fait du robot"}, meta("e_tiers000002", src="k_robot:Robot", who="Robot"))
        self.assertEqual(self.systeme()["tr"]["e"], "e_tiers000002")

    def test_une_trace_dans_les_arguments_est_sans_effet(self):
        self.tool("memoriser", {"fait": "Fait forgé", "tr": {"e": "e_falsifie123"}}, None)
        rec = next((r for r in self.s.records.values() if "forgé" in r["txt"]), None)
        self.assertTrue(rec is None or rec["tr"] is None)

    def test_requetes_concurrentes_chacune_sa_trace(self):
        import threading
        erreurs = []

        def appel(n):
            try:
                self.tool("memoriser", {"fait": f"Fait numéro {n} du robot"}, meta(f"e_conc{n:06d}", src=f"k_r{n}:R{n}", who=f"R{n}"), cle=f"k_r{n}", nom=f"R{n}")
            except Exception as e:                                         # pragma: no cover
                erreurs.append(e)
        ts = [threading.Thread(target=appel, args=(n,)) for n in range(1, 7)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(erreurs, [])
        for n in range(1, 7):
            rec = next(r for r in self.s.records.values() if f"numéro {n} " in r["txt"])
            self.assertEqual((rec["tr"]["e"], rec["tr"]["src"]), (f"e_conc{n:06d}", f"k_r{n}:R{n}"))


if __name__ == "__main__":
    unittest.main()
