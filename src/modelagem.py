#%%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
from pre_processamento import Engenharia_Features
from sklearn.model_selection import GridSearchCV



# %%
dados = pd.read_csv('../data/articles.csv')
# %%
dados.info()
contagem = dados["category"].value_counts()

categorias_validas = contagem[contagem >= 10].index

dados = dados[dados["category"].isin(categorias_validas)]
dados['text'].shape
# %%
features = ['title','text']
target = 'category'
X = dados[features]
y = dados[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# %%
preprocessador = ColumnTransformer(
    transformers=[
        (
            "texto",
            TfidfVectorizer(
                max_features=8000,
                ngram_range=(1,1),
                min_df=5
            ),
            "text_clean"
        ),
        (
            "titulo",
            TfidfVectorizer(
                max_features=3000,
                ngram_range=(1,2)
            ),
            "title_clean"
        ),
        (
            "numericas",
            StandardScaler(),
            [
                "text_qtd_palavras",
                "text_tamanho_medio_palavras",
                "title_qtd_palavras",
                "title_tamanho_medio_palavras"
            ]
        )
    ]
)
#%% Pipeline Simples 
modelo = LinearSVC(
      C=1.0,                  
    loss="squared_hinge",  
    dual=False,            
    class_weight="balanced",
    max_iter=5000,        
    tol=1e-3,
    random_state=42
)
pipeline = Pipeline([
    ("engenharia", Engenharia_Features()),
    ("preprocessamento", preprocessador),
    ("modelo", modelo)
])
pipeline
# %%
pipeline.fit(X_train, y_train)
pred1 = pipeline.predict(X_test)
#%%
print(f'Acurácia: {accuracy_score(y_test, pred1)}')

#%%
joblib.dump(pipeline, "../models/sgdc_classifier_084.joblib")

# %%
