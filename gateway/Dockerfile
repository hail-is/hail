FROM ubuntu:18.04

RUN echo "APT::Acquire::Retries \"3\";" > /etc/apt/apt.conf.d/80-retries && \
    apt-get update -y && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

RUN rm -f /etc/nginx/sites-enabled/default
ADD gateway.nginx.conf /etc/nginx/conf.d/gateway.conf
ADD gzip.conf /etc/nginx/conf.d/gzip.conf

RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

CMD ["nginx", "-g", "daemon off;"]
