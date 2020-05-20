#!/bin/bash

set -ex

openssl pkcs12 -export \
        -inkey /ssl-config/shuffler-key.pem \
        -in /ssl-config/shuffler-cert.pem \
        -name shuffler-key-store \
        -out shuffler-key-store.p12 \
        -passout pass:hail

openssl pkcs12 -export \
        -nokeys \
        -in /ssl-config/shuffler-incoming.pem \
        -out shuffler-trust-store.p12 \
        -passout pass:hail

java -jar /hail.jar is.hail.shuffler.server.ShuffleServer \
     shuffler-key-store.p12 \
     hail \
     shuffler-trust-store.p12 \
     hail \
     443
