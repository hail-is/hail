FROM {{ hail_ubuntu_image.image }}

RUN apt update && \
    apt install -y nginx wget certbot python3-certbot-nginx && \
    rm -rf /var/lib/apt/lists/*

RUN hail-pip-install cffi

RUN wget -O /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/v1.11.3/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl

RUN rm -f /etc/nginx/sites-enabled/default
ADD letsencrypt.nginx.conf /etc/nginx/conf.d/letsencrypt.conf

ADD letsencrypt.sh /

CMD ["/bin/bash", "/letsencrypt.sh"]
