FROM certbot/certbot

RUN wget -O /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/v1.11.3/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl
ADD letsencrypt.sh revoke-certs.sh /
