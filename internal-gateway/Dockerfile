FROM envoyproxy/envoy:v1.22.3

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
COPY internal-gateway-envoy.yaml /etc/envoy/envoy.yaml
RUN chmod go+r /etc/envoy/envoy.yaml
