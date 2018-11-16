FROM alpine:3.8

RUN apk update
RUN apk add python3 py3-cffi py3-cryptography
RUN pip3 install -U pip
RUN pip install flask
RUN pip install kubernetes
RUN pip install cerberus

COPY batch /batch
RUN pip install /batch

EXPOSE 5000

CMD ["python3", "-c", "import batch.server; batch.server.serve()"]
