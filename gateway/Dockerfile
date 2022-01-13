FROM {{ hail_ubuntu_image.image }}

RUN hail-apt-get-install nginx

RUN rm -f /etc/nginx/sites-enabled/default
ADD nginx.conf /etc/nginx/
ADD gateway.nginx.conf.out /etc/nginx/conf.d/gateway.conf
ADD gzip.conf /etc/nginx/conf.d/gzip.conf

RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

CMD ["nginx", "-g", "daemon off;"]
