FROM debian:9.5

RUN apt-get -y update && \
    apt-get -y install python python-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U \
    flask \
    google-api-python-client \
    google-auth google-auth-oauthlib \
    google-auth-httplib2 \
    requests

COPY upload /upload

EXPOSE 5000

CMD ["/bin/bash", "-c", "cd /upload && python main.py"]
