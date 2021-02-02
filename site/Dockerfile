FROM {{ hail_ubuntu_image.image }}

RUN hail-apt-get-install \
  nginx \
  curl \
  libnginx-mod-http-subs-filter

# link nginx logs to docker log collectors
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN rm -f /etc/nginx/sites-enabled/default && \
    rm -f /etc/nginx/nginx.conf
ADD hail.nginx.conf.in /
ADD nginx.conf /etc/nginx/

ADD site.sh /site.sh
ADD site-www.tar.gz /var/www/html
RUN chown -R www-data /var/www/html && \
    chmod -R u=rX,g=rX /var/www/html

CMD ["/bin/bash", "/site.sh"]
