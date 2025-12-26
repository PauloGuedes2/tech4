FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# ===============================
# Sistema e dependências básicas
# ===============================
RUN apt-get update && apt-get install -y \
    ca-certificates \
    tzdata \
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    libfontconfig1 \
    wget \
    curl \
    adduser \
    apt-utils \
    nginx \
    supervisor \
    sqlite3 \
 && rm -rf /var/lib/apt/lists/*

# ===============================
# Grafana (instalação via .deb)
# ===============================
RUN wget https://dl.grafana.com/oss/release/grafana_10.4.3_amd64.deb \
 && dpkg -i grafana_10.4.3_amd64.deb || true \
 && apt-get update \
 && apt-get -f install -y \
 && rm -rf /var/lib/apt/lists/* grafana_10.4.3_amd64.deb

# ===============================
# Prometheus
# ===============================
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.52.0/prometheus-2.52.0.linux-amd64.tar.gz \
 && tar -xzf prometheus-2.52.0.linux-amd64.tar.gz \
 && mv prometheus-2.52.0.linux-amd64/prometheus /usr/local/bin/ \
 && mv prometheus-2.52.0.linux-amd64/promtool /usr/local/bin/ \
 && mkdir -p /etc/prometheus \
 && rm -rf prometheus-2.52.0*

# ===============================
# Aplicação FastAPI
# ===============================
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/

# ===============================
# Configurações
# ===============================
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deploy/grafana.ini /etc/grafana/grafana.ini
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# ===============================
# Porta (Render ignora, mas documenta)
# ===============================
EXPOSE 10000

# ===============================
# Start
# ===============================
CMD ["/usr/bin/supervisord"]
