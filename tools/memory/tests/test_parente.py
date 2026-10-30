"""Tests du moteur de parenté : familles construites à la main, sans mémoire ni LLM."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import parente  # noqa: E402


def famille(*liens, sexes=None):
    """liens : (a, 'parent'|'enfant'|'conjoint'|'fratrie', b) -> « a est le <type> de b ». Les noms servent d'identités."""
    rels = [{"a": a, "l": t, "b": b, "type": t} for a, t, b in liens]
    sexes = sexes or {}
    return parente.construire(rels, lambda n: n.lower(), lambda i: sexes.get(i, ""))


SEXES = {"olivier": "m", "tiphaine": "f", "marion": "f", "emma": "f", "paul": "m", "luc": "m", "lea": "f",
         "jean": "m", "marie": "f", "anna": "f", "tom": "m", "zoe": "f"}


class LiensDeBase(unittest.TestCase):
    def setUp(self):
        # Le foyer réel : Olivier et Tiphaine sont mariés ; Marion et Emma sont les filles de Tiphaine ; Marion est la fille d'Olivier
        self.f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "parent", "marion"), ("tiphaine", "parent", "emma"),
                         ("olivier", "parent", "marion"), sexes=SEXES)

    def lien(self, a, b):
        return parente.lien(self.f, a, b)

    def test_mere_et_pere(self):
        self.assertEqual(self.lien("marion", "tiphaine"), "la mère")
        self.assertEqual(self.lien("marion", "olivier"), "le père")

    def test_enfant(self):
        self.assertEqual(self.lien("tiphaine", "marion"), "la fille")
        self.assertEqual(self.lien("olivier", "marion"), "la fille")

    def test_conjoints(self):
        self.assertEqual(self.lien("tiphaine", "olivier"), "le mari")
        self.assertEqual(self.lien("olivier", "tiphaine"), "la femme")

    def test_presomption_la_fille_du_couple_plutot_que_la_belle_fille(self):
        """Emma n'a que Tiphaine comme parent connu, et Tiphaine n'a qu'un conjoint (Olivier) : on présume qu'Emma est
        aussi sa fille. « fille » et « père » l'emportent sur « belle-fille » et « beau-père »."""
        self.assertEqual(self.lien("olivier", "emma"), "la fille")
        self.assertEqual(self.lien("emma", "olivier"), "le père")

    def test_presomption_la_soeur_plutot_que_la_demi_soeur(self):
        self.assertEqual(self.lien("marion", "emma"), "la sœur")
        self.assertEqual(self.lien("emma", "marion"), "la sœur")

    def test_declaration_demi_soeur_corrige_la_presomption(self):
        f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "parent", "marion"), ("tiphaine", "parent", "emma"),
                    ("olivier", "parent", "marion"), ("marion", "demi", "emma"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "marion", "emma"), "la demi-sœur")
        self.assertEqual(parente.lien(f, "emma", "marion"), "la demi-sœur")
        self.assertEqual(parente.lien(f, "olivier", "emma"), "la belle-fille")        # la déclaration retire aussi le père présumé

    def test_declaration_belle_fille_corrige_la_presomption(self):
        f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "parent", "emma"), ("emma", "beauenfant", "olivier"),
                    sexes=SEXES)
        self.assertEqual(parente.lien(f, "olivier", "emma"), "la belle-fille")
        self.assertEqual(parente.lien(f, "emma", "olivier"), "le beau-père")

    def test_declaration_beau_pere(self):
        f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "parent", "emma"), ("olivier", "beauparent", "emma"),
                    sexes=SEXES)
        self.assertEqual(parente.lien(f, "emma", "olivier"), "le beau-père")

    def test_pas_de_presomption_si_le_parent_a_plusieurs_conjoints(self):
        """Tiphaine a eu deux conjoints : impossible de savoir lequel est le père, donc on ne présume rien."""
        f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "conjoint", "paul"), ("tiphaine", "parent", "emma"),
                    sexes=SEXES)
        self.assertEqual(parente.lien(f, "emma", "olivier"), "le beau-père")
        self.assertEqual(parente.lien(f, "emma", "paul"), "le beau-père")

    def test_pas_de_presomption_si_deux_parents_connus(self):
        f = famille(("tiphaine", "conjoint", "olivier"), ("tiphaine", "parent", "emma"), ("paul", "parent", "emma"),
                    sexes=SEXES)
        self.assertEqual(parente.lien(f, "emma", "olivier"), "le beau-père")           # Paul est le père connu
        self.assertEqual(parente.lien(f, "emma", "paul"), "le père")

    def test_pas_de_presomption_sans_conjoint_connu(self):
        f = famille(("tiphaine", "parent", "emma"), ("tiphaine", "parent", "marion"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "marion", "emma"), "la sœur")                  # deux filles de la même mère : sœurs
        self.assertEqual(parente.lien(f, "emma", "olivier"), "")

    def test_pas_de_belle_fille_pour_sa_propre_fille(self):
        """Le défaut réel du 30/10 : Marion est la fille de Tiphaine, jamais sa belle-fille."""
        self.assertEqual(self.lien("tiphaine", "marion"), "la fille")
        self.assertNotIn("belle", self.lien("tiphaine", "marion"))

    def test_deux_vraies_soeurs_avec_les_deux_parents_communs(self):
        f = famille(("tiphaine", "parent", "marion"), ("olivier", "parent", "marion"),
                    ("tiphaine", "parent", "lea"), ("olivier", "parent", "lea"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "marion", "lea"), "la sœur")

    def test_inconnus(self):
        self.assertEqual(self.lien("marion", "inconnu"), "")


class GrandsParentsOnclesCousins(unittest.TestCase):
    def setUp(self):
        # jean+marie -> olivier, anna ; olivier -> marion ; anna -> tom ; tom -> zoe
        self.f = famille(("jean", "parent", "olivier"), ("marie", "parent", "olivier"),
                         ("jean", "parent", "anna"), ("marie", "parent", "anna"),
                         ("olivier", "parent", "marion"), ("anna", "parent", "tom"), ("tom", "parent", "zoe"),
                         ("jean", "conjoint", "marie"), sexes=SEXES)

    def lien(self, a, b):
        return parente.lien(self.f, a, b)

    def test_grands_parents(self):
        self.assertEqual(self.lien("marion", "jean"), "le grand-père")
        self.assertEqual(self.lien("marion", "marie"), "la grand-mère")

    def test_petits_enfants(self):
        self.assertEqual(self.lien("jean", "marion"), "la petite-fille")

    def test_oncle_tante_neveu_niece(self):
        self.assertEqual(self.lien("marion", "anna"), "la tante")
        self.assertEqual(self.lien("anna", "marion"), "la nièce")
        self.assertEqual(self.lien("tom", "olivier"), "l'oncle")
        self.assertEqual(self.lien("olivier", "tom"), "le neveu")

    def test_cousin_germain(self):
        self.assertEqual(self.lien("marion", "tom"), "le cousin germain")
        self.assertEqual(self.lien("tom", "marion"), "la cousine germaine")

    def test_cousin_issu_de_germain(self):
        """Deux enfants de cousins germains : même niveau (3, 3), cousins issus de germains."""
        f = famille(("jean", "parent", "olivier"), ("jean", "parent", "anna"),
                    ("olivier", "parent", "marion"), ("anna", "parent", "tom"),
                    ("marion", "parent", "lea"), ("tom", "parent", "zoe"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "lea", "zoe"), "la cousine issue de germain")
        self.assertEqual(parente.lien(f, "marion", "tom"), "le cousin germain")

    def test_grand_oncle_et_petite_niece(self):
        """Zoé (fille de Tom, petit-fils d'Anna) : Olivier, frère d'Anna, est son grand-oncle."""
        self.assertEqual(self.lien("zoe", "olivier"), "le grand-oncle")
        self.assertEqual(self.lien("olivier", "zoe"), "la petite-nièce")

    def test_generations_decalees_non_nommees(self):
        """Marion et Zoé : décalage de génération (2 et 3). Pas de nom courant : formulation prudente."""
        self.assertEqual(self.lien("marion", "zoe"), "une parente éloignée")

    def test_arriere_petite_fille(self):
        self.assertEqual(self.lien("anna", "zoe"), "la petite-fille")

    def test_arriere_grand_parent(self):
        f = famille(("jean", "parent", "olivier"), ("olivier", "parent", "marion"), ("marion", "parent", "zoe"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "zoe", "jean"), "l'arrière-grand-père")

    def test_pas_de_boucle_infinie(self):
        f = famille(("a", "parent", "b"), ("b", "parent", "c"), ("c", "parent", "a"))      # cycle absurde
        parente.lien(f, "a", "c")                                                        # doit simplement revenir


class SexeInconnu(unittest.TestCase):
    def test_formulations_neutres(self):
        f = famille(("p", "parent", "e"), ("p", "conjoint", "q"))
        self.assertEqual(parente.lien(f, "e", "p"), "un parent")
        self.assertEqual(parente.lien(f, "p", "e"), "un enfant")
        self.assertEqual(parente.lien(f, "p", "q"), "le conjoint ou la conjointe")

    def test_frere_ou_soeur_neutre(self):
        f = famille(("p", "parent", "a"), ("p", "parent", "b"))
        self.assertEqual(parente.lien(f, "a", "b"), "un frère ou une sœur")


class AllianceEtFratrieDeclaree(unittest.TestCase):
    def test_gendre_et_bru(self):
        f = famille(("olivier", "parent", "marion"), ("tom", "conjoint", "marion"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "olivier", "tom"), "le gendre")
        f = famille(("tiphaine", "parent", "paul"), ("lea", "conjoint", "paul"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "tiphaine", "lea"), "la bru")

    def test_beau_frere_et_belle_soeur(self):
        f = famille(("tiphaine", "parent", "marion"), ("tiphaine", "parent", "paul"), ("lea", "conjoint", "paul"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "marion", "lea"), "la belle-sœur")
        f = famille(("tiphaine", "parent", "marion"), ("olivier", "conjoint", "marion"), ("tiphaine", "parent", "paul"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "paul", "olivier"), "le beau-frère")

    def test_beau_pere_via_le_conjoint(self):
        f = famille(("jean", "parent", "olivier"), ("tiphaine", "conjoint", "olivier"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "tiphaine", "jean"), "le beau-père")

    def test_fratrie_declaree_sans_parents(self):
        f = famille(("marion", "fratrie", "emma"), sexes=SEXES)
        self.assertEqual(parente.lien(f, "marion", "emma"), "la sœur")


class Phrase(unittest.TestCase):
    def test_phrase(self):
        f = famille(("tiphaine", "parent", "marion"), sexes=SEXES)
        self.assertEqual(parente.phrase(f, "marion", "tiphaine", "Marion", "Tiphaine"), "Tiphaine est la mère de Marion")
        self.assertEqual(parente.phrase(f, "marion", "x", "Marion", "X"), "")

    def test_tous_les_liens(self):
        f = famille(("tiphaine", "parent", "marion"), ("olivier", "conjoint", "tiphaine"), sexes=SEXES)
        res = parente.tous_les_liens(f, "marion", ["marion", "tiphaine", "olivier", "inconnu"])
        self.assertEqual(res["tiphaine"], "la mère")
        self.assertNotIn("marion", res)
        self.assertNotIn("inconnu", res)


if __name__ == "__main__":
    unittest.main()
