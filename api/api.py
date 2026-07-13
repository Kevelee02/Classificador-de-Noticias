#%%
#%%
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
RESUMO_PATH = ROOT / "data" / "resumo_categorias.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from src.pre_processamento import Engenharia_Features

#%% Configuração da página — "wide" libera a largura total da tela
st.set_page_config(
    page_title="Classificador de Notícias",
    page_icon="📰",
    layout="wide"
)


#%% Carregar pipeline
MODELO_PATH = ROOT / "models" / "sgdc_classifier_084.joblib"

@st.cache_resource
def carregar_pipeline():
    return joblib.load(MODELO_PATH)

try:
    pipeline = carregar_pipeline()
    modelo_carregado = True
except FileNotFoundError:
    modelo_carregado = False
    st.error(f"Modelo não encontrado em {MODELO_PATH}.")


#%% Carregar top 10 categorias do dataset
@st.cache_data
#%% Carregar top 10 categorias (a partir de um resumo leve, não do CSV completo)


@st.cache_data
def carregar_top10_categorias():
    with open(RESUMO_PATH, "r", encoding="utf-8") as f:
        resumo = json.load(f)
    return pd.Series(resumo["top10_categorias"]).sort_values(ascending=False)


#%% Layout: duas colunas — classificador (maior, foco principal) | top 10 (menor)
col_principal, col_lateral = st.columns([2, 1], gap="large")

with col_principal:
    st.title("Classificador de Notícias")
    st.write("Digite o título e o conteúdo da notícia para prever sua categoria.")

    titulo = st.text_input("Título")
    texto = st.text_area("Texto da notícia", height=280)

    if st.button("Classificar", disabled=not modelo_carregado):

        if titulo.strip() == "" or texto.strip() == "":
            st.warning("Preencha o título e o texto.")
        else:
            entrada = pd.DataFrame({"title": [titulo], "text": [texto]})

            probabilidades = pipeline.predict_proba(entrada)[0]
            classes = pipeline.classes_

            top3_idx = np.argsort(probabilidades)[::-1][:3]
            top3_categorias = classes[top3_idx]
            top3_probs = probabilidades[top3_idx] * 100

            st.success(f"Categoria prevista: **{top3_categorias[0]}** "
                       f"({top3_probs[0]:.1f}% de confiança)")

            st.subheader("Top 3 categorias mais prováveis")

            cores_viridis = ["#440154", "#21918c", "#fde725"]
            medalhas = ["🥇", "🥈", "🥉"]

            for posicao, (categoria, prob) in enumerate(zip(top3_categorias, top3_probs)):
                cor = cores_viridis[posicao]
                largura = max(prob, 3)

                st.markdown(
                    f"""
                    <div style="margin-bottom: 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-size: 17px; font-weight: 600;">
                                {medalhas[posicao]} {categoria}
                            </span>
                            <span style="font-size: 17px; font-weight: 600; color: {cor};">
                                {prob:.1f}%
                            </span>
                        </div>
                        <div style="background-color: rgba(128,128,128,0.2); border-radius: 6px; height: 10px; margin-top: 6px;">
                            <div style="background-color: {cor}; width: {largura}%; height: 10px; border-radius: 6px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
with col_lateral:
    with st.container(border=True):
        st.markdown("##### 📊 Top 10 categorias")

        try:
            top10 = carregar_top10_categorias()
            maximo = top10.max()

            cores_viridis_10 = [
                "#440154", "#482878", "#3e4a89", "#31688e", "#26828e",
                "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
            ]

            for cor, (categoria, contagem_cat) in zip(cores_viridis_10, top10.items()):
                largura = (contagem_cat / maximo) * 100
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-size: 12px; font-weight: 600;">{categoria}</span>
                            <span style="font-size: 11px; color: {cor};">{contagem_cat:,}</span>
                        </div>
                        <div style="background-color: rgba(128,128,128,0.2); border-radius: 4px; height: 6px; margin-top: 3px;">
                            <div style="background-color: {cor}; width: {largura}%; height: 6px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except FileNotFoundError:
            st.info("Resumo de categorias não disponível.")