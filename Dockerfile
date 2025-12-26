FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# ===== Dependências do sistema =====
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    supervisor \
    sqlite3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ===== Grafana =====
RUN curl -fsSL https://packages.grafana.com/gpg.key | gpg --dearmor -o /usr/share/keyrings/grafana.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/grafana.gpg] https://packages.grafana.com/oss/deb stable main" \
    > /etc/apt/sources.list.d/grafana.list \
 && apt-get update \
 && apt-get install -y grafana \
 && rm -rf /var/lib/apt/lists/*

# ===== Prometheus =====
RUN curl -LO https://github.com/prometheus/prometheus/releases/download/v2.52.0/prometheus-2.52.0.linux-amd64.tar.gz \
 && tar -xzf prometheus-2.52.0.linux-amd64.tar.gz \
 && mv prometheus-2.52.0.linux-amd64/prometheus /usr/local/bin/ \
 && mv prometheus-2.52.0.linux-amd64/promtool /usr/local/bin/ \
 && mkdir -p /etc/prometheus \
 && rm -rf prometheus-2.52.0*

# ===== App =====
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/

# ===== Configs =====
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 10000

CMD ["/usr/bin/supervisord"]
