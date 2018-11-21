FROM alpine:3.8

RUN apk update
RUN apk add python3 py3-cffi py3-cryptography

COPY batch /batch/batch
COPY setup.py /batch/
RUN pip3 install --no-cache-dir /batch

EXPOSE 5000

CMD ["python3", "-c", "import batch.server; batch.server.serve()"]
