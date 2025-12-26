# Use a imagem Python base
FROM python:3.11-slim

# Instalar Nginx, Grafana e dependências
RUN apt-get update && apt-get install -y nginx wget gnupg2 curl \
    && mkdir -p /etc/apt/keyrings \
    && wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor > /etc/apt/keyrings/grafana.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" > /etc/apt/list.d/grafana.list \
    && apt-get update && apt-get install -y grafana \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências da API
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código e configurações
COPY src/ ./src/
COPY nginx.conf /etc/nginx/nginx.conf
# Se você tiver a pasta grafana com provisioning, descomente a linha abaixo:
# COPY grafana/ /etc/grafana/

# Variáveis de ambiente para o Grafana rodar atrás de subcaminho (/grafana)
ENV GF_SERVER_ROOT_URL=%(protocol)s://%(domain)s:%(http_port)s/grafana/
ENV GF_SERVER_SERVE_FROM_SUB_PATH=true
ENV GF_SECURITY_ADMIN_PASSWORD=admin

# Script para rodar os 3 processos juntos
RUN echo '#!/bin/bash\n\
nginx -g "daemon off;" & \n\
grafana-server --homepath /usr/share/grafana --config /etc/grafana/grafana.ini & \n\
uvicorn src.app.main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# O Render vai se conectar na porta 10000 do Nginx
EXPOSE 10000

CMD ["/app/start.sh"]