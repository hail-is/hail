FROM alpine:3.8

RUN apk update
RUN apk add python3 # python3=3.6.4-r1
RUN pip3 install -U pip
RUN pip install flask
RUN pip install kubernetes
RUN pip install cerberus

COPY batch.py /

EXPOSE 5000

CMD ["python3", "/batch.py"]
