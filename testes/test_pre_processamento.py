import importlib
import unittest
import pandas as pd

from src.pre_processamento import Engenharia_Features


class TestEngenhariaFetures(unittest.TestCase):
    def setUp(self):
        self.transformer = Engenharia_Features()

    def test_limpeza_texto_remove_pontuacao_e_urls(self):
        texto = "Olá, mundo! Visite https://exemplo.com e veja #teste"
        resultado = self.transformer.limpeza_texto(texto)

        self.assertIn("olá", resultado)
        self.assertIn("mundo", resultado)
        self.assertNotIn("https://", resultado)
        self.assertNotIn("#", resultado)
        self.assertNotIn("!", resultado)

    def test_quantidade_palavras_e_tamanho_medio(self):
        texto = "bom dia brasil"
        self.assertEqual(self.transformer.quantidade_palavras(texto), 3)
        self.assertEqual(self.transformer.tamanho_medio_palavra(texto), 4.0)

    def test_transform_cria_features_e_preenche_nulos(self):
        df = pd.DataFrame(
            {
                "title": ["Título exemplo", None],
                "text": ["Texto de teste", "Texto 2"],
            }
        )

        resultado = self.transformer.fit_transform(df)

        self.assertIn("title_qtd_palavras", resultado.columns)
        self.assertIn("text_qtd_palavras", resultado.columns)
        self.assertIn("title_tamanho_medio_palavras", resultado.columns)
        self.assertIn("text_tamanho_medio_palavras", resultado.columns)

        self.assertEqual(resultado.loc[0, "title"], "título exemplo")
        self.assertEqual(resultado.loc[1, "title"], "")
        self.assertEqual(resultado.loc[0, "text"], "texto de teste")
        self.assertEqual(resultado.loc[0, "title_qtd_palavras"], 2)
        self.assertEqual(resultado.loc[1, "text_qtd_palavras"], 2)

    def test_fit_raises_for_missing_columns(self):
        df = pd.DataFrame({"title": ["teste"]})

        with self.assertRaises(ValueError):
            Engenharia_Features().fit(df)

    def test_importa_classes_do_modulo(self):
        module = importlib.import_module("src.pre_processamento")

        self.assertTrue(hasattr(module, "Engenharia_Features"))
        self.assertTrue(hasattr(module, "EngenhariaFeatureNLTK"))
        self.assertTrue(hasattr(module, "EngenhariaFeatureSpacy"))

    def test_importa_classes_do_modulo_raiz(self):
        module = importlib.import_module("pre_processamento")

        self.assertTrue(hasattr(module, "Engenharia_Features"))
        self.assertTrue(hasattr(module, "EngenhariaFeatureSpacy"))
        self.assertTrue(hasattr(module, "Engenharia_Fetures"))


if __name__ == "__main__":
    unittest.main()
