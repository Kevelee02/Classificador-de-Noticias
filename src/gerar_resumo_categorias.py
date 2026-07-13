"""
Gera um resumo leve (top 10 categorias mais frequentes) a partir do
dataset completo, para ser usado pela API sem depender do articles.csv
inteiro (que é pesado demais para versionar no GitHub).

Rodar uma vez, sempre que o dataset for atualizado:
    python src/gerar_resumo_categorias.py
"""
#%%
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def gerar_resumo():
    dados = pd.read_csv(ROOT / "data" / "articles.csv")

    dados["category"] = dados["category"].replace(
        {"guia-de-livros-discos-filmes": "guia-de-livros-filmes-discos"}
    )

    top10 = (
        dados.groupby("category")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    resumo = {"top10_categorias": top10.to_dict()}

    caminho_saida = ROOT / "data" / "resumo_categorias.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"Resumo salvo em {caminho_saida}")
    print(top10)


if __name__ == "__main__":
    gerar_resumo()

# %%
