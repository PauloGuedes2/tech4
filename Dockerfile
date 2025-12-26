FROM grafana/grafana-oss:10.4.3

USER root

# ===============================
# Dependências do sistema (Alpine)
# ===============================
RUN apk update && apk add --no-cache \
    python3 \
    py3-pip \
    nginx \
    supervisor \
    sqlite \
    wget \
    curl

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
# App FastAPI
# ===============================
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/ ./src/

# ===============================
# Configs
# ===============================
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deploy/grafana.ini /etc/grafana/grafana.ini
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.yml

# ===============================
# Start
# ===============================
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
