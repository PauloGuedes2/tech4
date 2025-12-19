# Use uma imagem base Python oficial
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
# O arquivo requirements.txt deve estar na raiz do diretório tech4_feature_api
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
# O código está em 'src' dentro do diretório tech4_feature_api
COPY src/ ./src/

# Expõe a porta que o Uvicorn irá usar
EXPOSE 8000 3000

# Comando para rodar a aplicação com Uvicorn
# O módulo principal é 'src.app.main' e a instância do FastAPI é 'app'
# O host 0.0.0.0 é necessário para que o container seja acessível externamente
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
