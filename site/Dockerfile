FROM ubuntu:18.04

RUN apt-get update -y

# get add-apt-repository
RUN apt-get install -y software-properties-common
RUN bash -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata'

RUN add-apt-repository -y ppa:certbot/certbot
RUN apt-get update -y
RUN apt-get install -y python-certbot-nginx

RUN apt-get install -y nginx curl python-minimal rsyslog

RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    mv /root/google-cloud-sdk /
ENV PATH $PATH:/google-cloud-sdk/bin

RUN rm -f /etc/nginx/sites-enabled/default

ADD hail.nginx.conf /etc/nginx/conf.d/hail.conf

ADD poll-0.1.sh .
ADD poll-devel.sh .
ADD site.sh .

ADD poll-docs-crontab /etc/cron.d/
ADD poll-docs-cron.sh .

CMD ["bash", "/site.sh"]
