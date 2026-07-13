#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
#%%
dados = pd.read_csv('data/articles.csv',
                    parse_dates = ['date']
                    )
dados.head()
#%%
dados.info()
#%%
dados['fonte'] = dados['link'].str.extract(r"//([^/]+)")
df_fonte = dados.groupby('fonte').size().sort_values(ascending=False).reset_index()
print(df_fonte)
#%% [markdown]
#A esmagadora maioria das noticias vem apenas de uma fonte
#%% [markdown]# 
# ### Subcategoria
#
# A subcategoria não parece fazer muito sentido quando se trata de
# categorização o que parece ser mais sobre pessoas envolvidas na notícia
# ou o objeto de foco da notícia, e não uma subcategoria em si.
# Além dos muitos valores ausentes, isso torna a coluna pouco útil.
#%%

dados['subcategory'].loc[dados['subcategory'].notna()]
# %% [markdown]
#  ### Quantas categorias diferentes existem
#Algumas categorias não parecem se encaixar em um perfil claro, mas como
# não existe uma recomendação de remoção, a estratégia foi apenas tratar
# as categorias que são iguais, porém escritas de maneiras diferentes —
# por exemplo, `guia-de-livros-filmes-discos` e `guia-de-livros-discos-filmes`.
# %%

print(dados['category'].unique())
df_categoria = dados.groupby('category').size().sort_values(ascending=False).reset_index()

# %%
df_categoria = df_categoria.rename(columns={0:'Qtd_Noticias'})
# %%
dados['category'] = dados['category'].replace(
    {'guia-de-livros-discos-filmes': 'guia-de-livros-filmes-discos'}
)
print(dados.groupby('category').size())
# %% Top 10 Categorias que mais aparece
top_9 = df_categoria.iloc[:9,:]
soma_outros = df_categoria.iloc[9:,1].sum()
outros = pd.DataFrame({
    'category': ['Outros'],
    'Qtd_Noticias': [soma_outros]
})
print(outros)
top_10 = pd.concat([top_9, outros], ignore_index=True)
top_10 = top_10.sort_values(by='Qtd_Noticias', ascending=False)
top_10
# %%
plt.figure(figsize=(10,10))
sns.set_theme(style="whitegrid") 

ax = sns.barplot(
    y=top_10['category'], 
    x=top_10['Qtd_Noticias'], 
    palette='viridis',
)
plt.title('Top 10 Categorias de Notícias', fontsize=16, pad=15)
plt.xlabel('Quantidade de Notícias', fontsize=12)
plt.ylabel('Categoria', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', 
                xytext=(5, 0), textcoords='offset points', fontsize=11)

plt.tight_layout()
plt.show()
#%% [markdown]
# A maioria das notícias fica concentrada no top 7, muito provavelmente
# porque são categorias mais abrangentes, onde é possível abordar os mais
# variados assuntos. Já as outras categorias são muito específicas —
# algumas chegando a receber apenas uma notícia
# %% [markdown]
# ### Bottom 5 categorias
last_5 = df_categoria.iloc[-6:-1,:]
plt.figure(figsize=(10,10))
sns.set_theme(style="whitegrid") 

ax = sns.barplot(
    y=last_5['category'], 
    x=last_5['Qtd_Noticias'], 
    palette='viridis',
)
plt.title('Bottom 5 Categorias de Notícias', fontsize=16, pad=15)
plt.xlabel('Quantidade de Notícias', fontsize=12)
plt.ylabel('Categoria', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', 
                xytext=(5, 0), textcoords='offset points', fontsize=11)

plt.tight_layout()
plt.show()
#%% [markdown]
#As últimas 5 categorias existem pouquisssimas noticias, algumas tendo apenas 1 registro
# %%
dados.info()
# %% Analisando o tamanho dos textos
dados['tamanho_noticia'] = dados['text'].apply(
    lambda texto: len(str(texto).split())
    )
dados['tamanho_titulo']=dados['title'].apply(
    lambda title: len(str(title).split())
    )
# %%
dados.head()
# %%
dados['tamanho_noticia'].describe()
#%%
dados['tamanho_titulo'].describe()
# %%
sns.histplot(
    data= dados,
    x = dados['tamanho_noticia'],
    fill = True,
    palette= 'viridis'
)
plt.xlim(0, 2500)
plt.title('Distribuição da quantidade de palavras por Noticia')
plt.xlabel('Quantidade de Palavras')
plt.tight_layout()
plt.show()
#%% [markdown]
# Quando limitamos o eixo de 0 a 2500, é possível ver uma distribuição
# assimétrica à direita, com uma cauda longa. **75% das notícias têm entre
# 0 e 563 palavras**, com base no terceiro quartil. Isso pode facilitar a
# escolha do modelo, caso essa coluna entre como variável.
# %%
sns.boxplot(
    data = dados,
    y = dados['tamanho_noticia'],
    fill = True
)
plt.show()
# %%
plt.figure(figsize=(10,10))

categorias_mais_populares = ['poder','colunas','mercado','esporte',	
	'mundo','cotidiano','ilustrada','opiniao','paineldoleitor']
dados_grafico = dados.loc[dados['category'].isin(categorias_mais_populares)]
sns.boxplot(
    data = dados_grafico,
    y = 'tamanho_noticia',
    x = 'category',
    palette= 'viridis'
)
plt.ticklabel_format()
plt.show()
#%% [markdown]
# ### Distribuição de tamanho por categoria
#
# As categorias apresentam distribuições de tamanho semelhantes, com
# medianas e dispersões próximas. Todas possuem uma quantidade considerável
# de outliers, indicando que textos muito longos ocorrem em diferentes
# categorias e não parecem estar associados a uma categoria específica.
#
# - **`paineldoleitor`** concentra textos um pouco menores (caixa mais à esquerda)
# - **`mercado`** e **`poder`** possuem alguns dos maiores outliers, chegando
#   próximo de 10 mil palavras
# - As medianas de praticamente todas as categorias ficam na faixa de
#   **300–500 palavras**, reforçando que o comprimento típico das notícias
#   é parecido
# %%
dados_grafico.groupby('category')['tamanho_noticia'].agg([
    'count',
    'median',
    'mean',
    'std',
    'min',
    'max'
]).sort_values('median', ascending=False)
# %% Olhando para o tamaho do título por categoria
plt.figure(figsize=(10,10))

categorias_mais_populares = ['poder','colunas','mercado','esporte',	
	'mundo','cotidiano','ilustrada','opiniao','paineldoleitor']
dados_grafico = dados.loc[dados['category'].isin(categorias_mais_populares)]
sns.boxplot(
    data = dados_grafico,
    y = 'tamanho_titulo',
    x = 'category',
    palette= 'viridis'
)
plt.show()
#%% [markdown]
#Ao analisar o boxplot do tamanho dos títulos por categoria, observa-se
# que a maioria das editorias apresenta distribuições semelhantes, com
# medianas entre 10 e 11 palavras e dispersão relativamente próxima.
#
# Entretanto, algumas categorias se destacam:
# - **`opiniao`** possui os títulos mais curtos, com mediana em torno de
#   4 palavras
# - **`colunas`** também apresenta títulos menores e maior variabilidade
# - **`esporte`**, **`cotidiano`**, **`mercado`**, **`mundo`** e **`poder`**
#   mantêm títulos mais longos e consistentes, sugerindo um padrão de
#   título semelhante entre elas
# %% [markdown]
# ### WordClou das noticias
#%%





#%%

stopwords = STOPWORDS.union({
    "pra", "vai", "ser", "sobre",
    "ainda", "disse", "ter", "também",
    "já", "porque", "segundo",
    "e", "de", "da", "do", "dos", "das", "em", "no", "na", "nos", "nas",
    "para", "por", "um", "uma", "uns", "umas", "o", "a", "os", "as",
    "que", "como", "com", "não", "se", "é", "ao", "à", "à", "são",
    "uol", "disse", "afirmou", "segundo", "foto","folha", "brasil","brasileiro", "anos",
    "ano", 'foi', 'mai', 'sua', 'entre','seu','ou','diz' ,'tem','quando'
})

texto = " ".join(dados["text"].dropna())
#%%

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=200,
    stopwords= stopwords,
    collocations=False
).generate(texto)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud das notícias")
plt.show()
# %%
dados['year_date'] = dados['date'].dt.year
dados['year_date'].unique()
# %%
noticias_por_ano = (
    dados['year_date']
    .value_counts()
    .sort_index()
)
noticias_por_ano
# %%
plt.figure(figsize=(10,10))

sns.barplot(
    x=noticias_por_ano.index,
    y=noticias_por_ano.values,
    palette= 'viridis'
)
plt.title('Quantidade de notícias por ano')
plt.xlabel('Ano')
plt.ylabel('Número de notícias')

plt.show()
# %% Categoria mais popular por ano
categoria_ano = (
    dados
    .groupby(['year_date', 'category'])
    .size()
    .reset_index(name='quantidade')
)
categoria_mais_popular = (
    categoria_ano
    .sort_values('quantidade', ascending=False)
    .groupby('year_date')
    .first()
    .reset_index()
)

categoria_mais_popular

# %% [markdown]
# É possível ver que, ao longo dos anos, a quantidade de notícias foi se
# tornando cada vez menor. Um fator interessante é que, em cada ano, a
# categoria mais popular foi diferente.
# %%