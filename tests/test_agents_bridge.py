"""Pont entre boucles asyncio (components/agents/bridge.py) : un générateur s'exécute DANS la boucle de l'interface et rend ses éléments à la boucle de l'API.
Deux vraies boucles, dans deux threads. Bibliothèque standard seulement."""
import asyncio
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.agents import bridge  # noqa: E402


class BoucleEnThread:
    """Une boucle asyncio qui tourne dans son propre thread, comme la boucle de Gradio."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.fil = threading.Thread(target=self._run, daemon=True)
        self.fil.start()
        self.pret = threading.Event()
        self.loop.call_soon_threadsafe(self.pret.set)
        self.pret.wait(2)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def arreter(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.fil.join(2)


class PontTests(unittest.TestCase):
    def setUp(self):
        self.interface = BoucleEnThread()
        self.holder = bridge.LoopHolder()
        self.holder.set(self.interface.loop)
        self.fils = []

    def tearDown(self):
        self.interface.arreter()

    def generateur(self, n=3, delai=0.0, erreur_a=None):
        async def gen():
            self.fils.append(threading.current_thread().name)
            try:
                for i in range(n):
                    if erreur_a == i:
                        raise RuntimeError("panne dans le générateur")
                    if delai:
                        await asyncio.sleep(delai)
                    yield i
            finally:
                self.ferme = True
        self.ferme = False
        return gen

    def collecter(self, factory, holder=None):
        async def principal():
            return [x async for x in bridge.run_in_loop(holder or self.holder, factory)]
        return asyncio.run(principal())

    def test_les_elements_arrivent_dans_l_ordre(self):
        self.assertEqual(self.collecter(self.generateur(5)), [0, 1, 2, 3, 4])

    def test_le_generateur_s_execute_dans_la_boucle_de_l_interface(self):
        self.collecter(self.generateur())
        self.assertEqual(self.fils, [self.interface.fil.name])
        self.assertNotEqual(self.fils[0], threading.current_thread().name)

    def test_la_boucle_de_l_appelant_reste_libre_pendant_le_tour(self):
        battements = []

        async def principal():
            async def battement():
                for _ in range(10):
                    battements.append(1)
                    await asyncio.sleep(0.01)
            tache = asyncio.ensure_future(battement())
            sortie = [x async for x in bridge.run_in_loop(self.holder, self.generateur(4, delai=0.05))]
            await tache
            return sortie
        self.assertEqual(asyncio.run(principal()), [0, 1, 2, 3])
        self.assertEqual(len(battements), 10)

    def test_une_erreur_du_generateur_est_relevee_chez_l_appelant(self):
        with self.assertRaises(RuntimeError) as cm:
            self.collecter(self.generateur(5, erreur_a=2))
        self.assertIn("panne", str(cm.exception))
        self.assertTrue(self.ferme)

    def test_les_elements_avant_l_erreur_sont_livres(self):
        recus = []

        async def principal():
            try:
                async for x in bridge.run_in_loop(self.holder, self.generateur(5, erreur_a=2)):
                    recus.append(x)
            except RuntimeError:
                pass
        asyncio.run(principal())
        self.assertEqual(recus, [0, 1])

    def test_l_arret_de_l_appelant_ferme_le_generateur_d_origine(self):
        async def principal():
            gen = bridge.run_in_loop(self.holder, self.generateur(1000, delai=0.01))
            premier = await gen.__anext__()
            await gen.aclose()
            return premier
        self.assertEqual(asyncio.run(principal()), 0)
        for _ in range(100):
            if self.ferme:
                break
            time.sleep(0.02)
        self.assertTrue(self.ferme)                                                # le `finally` du générateur d'origine s'est exécuté

    def test_l_annulation_de_l_appelant_arrete_le_travail(self):
        async def principal():
            async def lire():
                async for _ in bridge.run_in_loop(self.holder, self.generateur(1000, delai=0.01)):
                    pass
            t = asyncio.ensure_future(lire())
            await asyncio.sleep(0.1)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        asyncio.run(principal())
        for _ in range(100):
            if self.ferme:
                break
            time.sleep(0.02)
        self.assertTrue(self.ferme)

    def travail_continue(self, arret_par):
        """Éléments produits APRÈS l'arrêt de l'appelant : un générateur qui continue (un LLM qui répond dans le vide, des outils appelés pour rien).
        La boucle de l'appelant reste VIVANTE pendant la mesure : sinon `call_soon_threadsafe` échouerait (« boucle fermée ») et arrêterait le générateur par accident,
        sans que l'arrêt voulu du pont n'y soit pour rien."""
        compte = {"n": 0}
        appelant = BoucleEnThread()

        async def gen():
            for i in range(10_000):
                compte["n"] = i
                await asyncio.sleep(0.005)
                yield i

        async def scenario():
            if arret_par == "aclose":
                g = bridge.run_in_loop(self.holder, gen)
                await g.__anext__()
                await g.__anext__()
                await g.aclose()
            else:
                async def lire():
                    async for _ in bridge.run_in_loop(self.holder, gen):
                        pass
                t = asyncio.ensure_future(lire())
                await asyncio.sleep(0.05)
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            return compte["n"]
        try:
            au_moment_de_l_arret = asyncio.run_coroutine_threadsafe(scenario(), appelant.loop).result(10)
            time.sleep(0.5)                                                       # laisse à un générateur qui ne s'arrête pas le temps de continuer
            return compte["n"] - au_moment_de_l_arret
        finally:
            appelant.arreter()

    def test_le_travail_s_arrete_quand_l_appelant_part_par_aclose(self):
        self.assertLessEqual(self.travail_continue("aclose"), 2)

    def test_le_travail_s_arrete_quand_l_appelant_est_annule(self):
        self.assertLessEqual(self.travail_continue("cancel"), 2)

    def test_sans_boucle_d_interface_le_generateur_est_itere_sur_place(self):
        vide = bridge.LoopHolder()
        self.assertEqual(self.collecter(self.generateur(3), holder=vide), [0, 1, 2])
        self.assertEqual(self.fils, [threading.current_thread().name])

    def test_si_l_appelant_est_deja_dans_la_boucle_de_l_interface_pas_de_pont(self):
        async def principal():
            h = bridge.LoopHolder()
            h.set(asyncio.get_running_loop())
            return [x async for x in bridge.run_in_loop(h, self.generateur(3))]
        self.assertEqual(asyncio.run(principal()), [0, 1, 2])

    def test_une_boucle_d_interface_arretee_est_ignoree(self):
        morte = asyncio.new_event_loop()
        h = bridge.LoopHolder()
        h.set(morte)
        self.assertEqual(self.collecter(self.generateur(2), holder=h), [0, 1])
        morte.close()

    def test_plusieurs_appelants_simultanes_ne_se_melangent_pas(self):
        async def un(n):
            async def gen():
                for i in range(4):
                    await asyncio.sleep(0.005 * (n % 3 + 1))
                    yield (n, i)
            return [x async for x in bridge.run_in_loop(self.holder, gen)]

        async def tous():
            return await asyncio.gather(*(un(n) for n in range(8)))
        resultats = asyncio.run(tous())
        for n, r in enumerate(resultats):
            self.assertEqual(r, [(n, i) for i in range(4)])

    def test_un_generateur_vide(self):
        async def vide():
            return
            yield
        self.assertEqual(self.collecter(vide), [])

    def test_les_exceptions_de_type_base_sont_aussi_relevees(self):
        class Sortie(BaseException):
            pass

        async def gen():
            yield 1
            raise Sortie()
        with self.assertRaises(Sortie):
            self.collecter(gen)


class HolderTests(unittest.TestCase):
    def test_get_set(self):
        h = bridge.LoopHolder()
        self.assertIsNone(h.get())
        loop = asyncio.new_event_loop()
        h.set(loop)
        self.assertIs(h.get(), loop)
        loop.close()


if __name__ == "__main__":
    unittest.main()
