#!/bin/bash
# Credit: https://gist.github.com/jessedearing/2351836
echo "Generating an SSL private key to sign your certificate..."
openssl genrsa -des3 -out myssl.key 1024

echo "Generating a Certificate Signing Request..."
openssl req -new -key myssl.key -out myssl.csr

echo "Removing passphrase from key (for nginx)..."
cp myssl.key myssl.key.org
openssl rsa -in myssl.key.org -out myssl.key
rm myssl.key.org

echo "Generating certificate..."
openssl x509 -req -days 365 -in myssl.csr -signkey myssl.key -out myssl.crt

echo "Copying certificate (myssl.crt) to /etc/ssl/certs/"
mkdir -p  /etc/ssl/certs
cp myssl.crt /etc/ssl/certs/

echo "Copying key (myssl.key) to /etc/ssl/private/"
mkdir -p  /etc/ssl/private
cp myssl.key /etc/ssl/private/

echo "Cleaning up"
rm myssl*