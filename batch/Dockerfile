FROM alpine:3.8

RUN apk add --no-cache \
  py3-cffi \
  py3-cryptography \
  python3 \
  && true

COPY setup.py /batch/
COPY batch /batch/batch/
RUN pip3 install --no-cache-dir /batch

EXPOSE 5000

CMD ["python3", "-c", "import batch.server; batch.server.serve()"]
