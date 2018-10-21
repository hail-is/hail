FROM debian:9.5

RUN apt-get update -y && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# link nginx logs to docker log collectors
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN rm -f /etc/nginx/sites-enabled/default
ADD hail.nginx.conf /etc/nginx/conf.d/hail.conf

CMD ["nginx", "-g", "daemon off;"]
