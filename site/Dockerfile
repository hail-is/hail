FROM debian:9.5

RUN apt-get update && \
    apt-get install -y nginx curl python-minimal && \
    rm -rf /var/lib/apt/lists/*

RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    mv /root/google-cloud-sdk /
ENV PATH $PATH:/google-cloud-sdk/bin

# link nginx logs to docker log collectors
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN rm -f /etc/nginx/sites-enabled/default && \
    rm -f /etc/nginx/nginx.conf
ADD hail.nginx.conf.in /
ADD nginx.conf /etc/nginx/

ADD poll.sh /poll.sh
ADD poll-0.1.sh /poll-0.1.sh
ADD poll-0.2.sh /poll-0.2.sh
ADD site.sh /site.sh
ADD site-www.tar.gz /var/www/html
RUN chown -R www-data /var/www/html && \
    chmod -R u=rX,g=rX /var/www/html

CMD ["/bin/bash", "/site.sh"]
