FROM debian:18.04

RUN apt-get update -y && \
    apt-get install -y nginx curl && \
    rm -rf /var/lib/apt/lists/*

RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    mv /root/google-cloud-sdk /
ENV PATH $PATH:/google-cloud-sdk/bin

# link nginx logs to docker log collectors
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN rm -f /etc/nginx/sites-enabled/default
ADD hail.is.nginx.conf /etc/nginx/conf.d/hail.is.conf

CMD ["nginx", "-g", "daemon off;"]
