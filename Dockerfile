# Imagem base com a mesma versão de Python usada no ambiente de
# desenvolvimento local (.venv), para evitar qualquer incompatibilidade
# sutil entre o modelo treinado localmente e o ambiente de execução.
FROM python:3.14-slim

# Evita que o Python gere arquivos .pyc e força output sem buffer (logs aparecem em tempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copia primeiro só o requirements.txt para aproveitar cache de camadas do Docker:
# se o código mudar mas as dependências não, o Docker reaproveita esta camada
# e não reinstala tudo do zero a cada build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código-fonte
COPY api/ api/
COPY src/ src/

# Modelo treinado (.joblib) — leve o suficiente para versionar no Git,
# então já vai embutido na imagem, sem precisar de volume em runtime.
COPY models/ models/

# Apenas o resumo pré-calculado das categorias (poucos KB), não o
# articles.csv completo (esse continua fora do repositório/imagem,
# usado só localmente para EDA e treino).
COPY data/resumo_categorias.json data/resumo_categorias.json

# Porta padrão que o Streamlit expõe
EXPOSE 8501

# Healthcheck simples: o Docker considera o container "saudável" quando o
# Streamlit responde na porta 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando padrão ao rodar o container
CMD ["streamlit", "run", "api/api.py", "--server.port=8501", "--server.address=0.0.0.0"]