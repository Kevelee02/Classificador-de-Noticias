# Classe de Engenharia de Features

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Engenharia_Features(BaseEstimator, TransformerMixin):
    """Realiza a limpeza de texto e cria features textuais para modelos."""

    def __init__(self, text_columns=None, fill_value=""):
        self.text_columns = ["title", "text"] if text_columns is None else list(text_columns)
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = self.confirmando_dataframe(X)
        missing_cols = [col for col in self.text_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Colunas ausentes para limpeza de texto: {missing_cols}")
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Aplicando todas as transformações e funções."""
        X = self.confirmando_dataframe(X)
        X = X.copy()

        for col in self.text_columns:
            if col not in X.columns:
                X[col] = ""
            X[col] = X[col].fillna(self.fill_value)
            X[col] = X[col].apply(self.limpeza_texto)

            X[f"{col}_clean"] = X[col]
            X[f"{col}_qtd_palavras"] = X[col].apply(self.quantidade_palavras)
            X[f"{col}_qtd_caracteres"] = X[col].apply(len)
            X[f"{col}_tamanho_medio_palavras"] = X[col].apply(self.tamanho_medio_palavra)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def confirmando_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def limpeza_texto(self, text):
        """Limpando o texto, removendo pontuações e caracteres especiais."""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+", " ", text)
        text = re.sub(r"[^\w\sáàâãéèêíïóôõöúçñ]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def quantidade_palavras(self, text):
        """Contando a quantidade de palavras por título/notícia."""
        return len(str(text).split()) if str(text).strip() else 0

    def tamanho_medio_palavra(self, text):
        """Calculando o tamanho médio das palavras."""
        words = str(text).split()
        if not words:
            return 0.0
        average = sum(len(word) for word in words) / len(words)
        return round(average, 2)

__all__ = [
    "Engenharia_Features",
]

    