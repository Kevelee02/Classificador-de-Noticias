# Classificador de Categorias de Notícias

Classificador de notícias de jornais brasileiros por categoria (esporte,
mercado, poder, mundo, etc.), com pipeline de NLP completo — da análise
exploratória até uma API funcional de classificação.

---


## Estrutura do projeto

```
CASE_TECNICO_NLP2/
├── api/
│   └── api.py                  # Interface Streamlit + endpoint de classificação
├── src/
│   ├── EDA.py                  # Análise exploratória dos dados
│   ├── pre_processamento.py    # Classe de engenharia de features (fonte única)
│   └── modelagem.py            # Treino, avaliação e serialização do modelo
├── testes/
│   ├── conftest.py
│   └── test_pre_processamento.py
├── data/
│   └── articles.csv            # Dataset de notícias
├── models/
│   └── sgdc_classifier_084.joblib   # Pipeline treinada (modelo final)
├── requirements.txt
├── README.md
└── pre_processamento.py
```

---

##  Dataset

Notícias de jornais brasileiros, com as colunas:

| Coluna | Descrição |
|---|---|
| `title` | Título da notícia |
| `text` | Corpo do texto |
| `date` | Data de publicação |
| `category` | Categoria (variável alvo) |
| `subcategory` | Subcategoria (muitos valores ausentes, descartada da modelagem) |
| `link` | URL da notícia (usada para extrair a fonte) |

---

##  Principais achados da EDA (`src/EDA.py`)

- A esmagadora maioria das notícias vem de uma única fonte
- A coluna `subcategory` tem muitos valores ausentes e pouca relação com uma
  hierarquia real de categoria — descartada da modelagem
- Após unificar grafias diferentes da mesma categoria (ex:
  `guia-de-livros-discos-filmes` vs. `guia-de-livros-filmes-discos`), as
  primeiras 7 categorias concentram a maior parte das notícias; as demais
  são bem mais específicas, algumas com pouquíssimos registros
- 75% das notícias têm entre 0 e 563 palavras (terceiro quartil); a
  distribuição de tamanho é assimétrica à direita, com cauda longa de textos
  bem mais extensos
- A distribuição de tamanho de texto e de título é similar entre as
  categorias mais populares, com exceção de `opiniao` e `colunas`, que têm
  títulos consideravelmente mais curtos
- A quantidade de notícias por ano vem diminuindo ao longo do tempo, e a
  categoria mais popular varia de ano para ano

---

##  Engenharia de features (`src/pre_processamento.py`)

Uma única classe `Engenharia_Features` (compatível com a API do
scikit-learn, via `BaseEstimator`/`TransformerMixin`) é usada tanto no
treino (`modelagem.py`) quanto na inferência (`api.py`), evitando
divergência entre as duas etapas. Ela:

- Normaliza o texto (minúsculas, remove URLs, menções/hashtags, pontuação)
- Gera features numéricas complementares ao texto: quantidade de palavras,
  quantidade de caracteres e tamanho médio das palavras (para título e
  corpo da notícia)


---

##  Modelagem (`src/modelagem.py`)

**Pipeline:** `Engenharia_Features` → `TfidfVectorizer` (texto e título,
unigramas + bigramas) + `StandardScaler` (features numéricas) →
classificador linear.

### Decisões e iterações

| Modelo testado | Observação |
|---|---|
| `LogisticRegression` (solver `lbfgs`) | ~84% de acurácia, porém treino levava 30+ minutos com as 167 mil linhas do dataset — inviável para iteração rápida |
| `LogisticRegression` (solver `saga`) | Ainda lento nessa escala (solvers full-batch escalam mal com muitas amostras/classes) |
| **`SGDClassifier` (loss `log_loss`)** | **Escolha final.** Treina por gradiente estocástico, levando menos de 5 minutos, com acurácia final de **83%** — perda de apenas 1 ponto frente ao modelo mais lento |

A escolha final prioriza um pipeline que permite iteração rápida (testar
hiperparâmetros, comparar variantes de features) sem sacrificar
praticamente nada de desempenho — trade-off importante dado o critério de
avaliação priorizar entrega funcional sobre resultado ótimo do modelo.

O modelo final expõe `.predict_proba()`, usado na API para mostrar as 3
categorias mais prováveis de cada notícia, não só a predição top-1.

---

##  API (`api/api.py`)

Interface construída em **Streamlit**, com:

- Campo de título e texto da notícia
- Botão de classificação, retornando a categoria prevista
- **Top 3 categorias mais prováveis**, com percentual de confiança
- Painel lateral com as **10 categorias mais frequentes** do dataset de
  treino, para contexto

### Como rodar


#### Opção 1 — Localmente (venv)
 
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar a API
streamlit run api/api.py
```
 
#### Opção 2 — Via Docker (recomendado para simular um ambiente de produção)
 
O modelo já treinado (`.joblib`) e o resumo de categorias vêm embutidos na
própria imagem
 
```bash
# 1. Construir a imagem
docker build -t classificador-noticias .
 
# 2. Rodar o container
docker run -p 8501:8501 classificador-noticias
```
 
Em ambos os casos, a API fica disponível em `http://localhost:8501`.

## Testes (`testes/`)

Testes unitários para a classe `Engenharia_Features`, cobrindo a limpeza de
texto e a geração das features numéricas.

```bash
pytest testes/
```

---

##  Tecnologias

- **Python 3.14**
- **pandas / numpy** — manipulação de dados
- **scikit-learn** — pipeline, TF-IDF, modelo de classificação
- **matplotlib / seaborn / wordcloud** — visualização na EDA
- **Streamlit** — interface da API
- **joblib** — serialização do pipeline treinado
- **pytest** — testes unitários

---
