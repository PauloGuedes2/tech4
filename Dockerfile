FROM python:3.11-slim

# Instalar Nginx, Grafana e dependências
# Correção: O caminho correto é /etc/apt/sources.list.d/
RUN apt-get update && apt-get install -y nginx wget gnupg2 curl \
    && mkdir -p /etc/apt/keyrings \
    && mkdir -p /etc/apt/sources.list.d \
    && wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor > /etc/apt/keyrings/grafana.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" > /etc/apt/sources.list.d/grafana.list \
    && apt-get update && apt-get install -y grafana \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e instala as dependências da API
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY src/ ./src/
COPY nginx.conf /etc/nginx/nginx.conf

# Configurações de subcaminho para o Grafana funcionar com Nginx
ENV GF_SERVER_ROOT_URL=%(protocol)s://%(domain)s:%(http_port)s/grafana/
ENV GF_SERVER_SERVE_FROM_SUB_PATH=true

# Script de inicialização (mantendo o Uvicorn na 8000 e Grafana na 3000)
RUN echo '#!/bin/bash\n\
grafana-server --homepath /usr/share/grafana --config /etc/grafana/grafana.ini & \n\
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 & \n\
nginx -g "daemon off;" \n\
' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 10000

CMD ["/app/start.sh"]