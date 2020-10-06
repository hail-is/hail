FROM ubuntu:18.04

RUN apt-get update -y

# get add-apt-repository
RUN echo "APT::Acquire::Retries \"3\";" > /etc/apt/apt.conf.d/80-retries && \
    apt-get update -y && \
    apt-get install -y nginx software-properties-common && \
    bash -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata' && \
    add-apt-repository -y ppa:certbot/certbot && \
    apt-get update -y && \
    apt-get install -y python-certbot-nginx wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/v1.11.3/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl

RUN rm -f /etc/nginx/sites-enabled/default
ADD letsencrypt.nginx.conf /etc/nginx/conf.d/letsencrypt.conf

ADD letsencrypt.sh /

CMD ["/bin/bash", "/letsencrypt.sh"]
