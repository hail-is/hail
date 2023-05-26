#!/bin/bash

if [ -z "${NAMESPACE}" ]; then
    echo "Must specify a NAMESPACE environment variable"
    exit 1;
elif [ "${NAMESPACE}" == "default" ]; then
    echo "This script is only for creating test database configs"
    exit 1;
fi

function create_key_and_cert() {
    local name=$1
    local key_file=${name}-key.pem
    local csr_file=${name}-csr.csr
    local cert_file=${name}-cert.pem

    openssl genrsa -out ${key_file} 4096
    openssl req -new -subj /CN=$name -key $key_file -out $csr_file
    openssl x509 -req -in $csr_file \
        -CA server-ca.pem -CAkey server-ca-key.pem \
        -CAcreateserial \
        -out $cert_file \
        -days 365 -sha256
}

dir=$(mktemp -d)
cd $dir

# Create the MySQL server CA
openssl req -new -x509 \
    -subj /CN=db-root -nodes -newkey rsa:4096 \
    -keyout server-ca-key.pem -out server-ca.pem

create_key_and_cert server
create_key_and_cert client

set +x
LC_ALL=C tr -dc '[:alnum:]' </dev/urandom | head -c 16 > db-root-password
password=$(cat db-root-password)

cat >sql-config.cnf <<EOF
[client]
host=db.$NAMESPACE
user=root
port=3306
password=$password
ssl-ca=/sql-config/server-ca.pem
ssl-cert=/sql-config/client-cert.pem
ssl-key=/sql-config/client-key.pem
ssl-mode=VERIFY_CA
EOF

cat >sql-config.json <<EOF
{
    "host": "db.$NAMESPACE.svc.cluster.local",
    "port": 3306,
    "user": "root",
    "password": "$password",
    "ssl-ca": "/sql-config/server-ca.pem",
    "ssl-cert": "/sql-config/client-cert.pem",
    "ssl-key": "/sql-config/client-key.pem",
    "ssl-mode": "VERIFY_CA",
    "instance": "dummy",
    "connection_name": "dummy"
}
EOF

kubectl create secret generic database-server-config \
    --namespace=$NAMESPACE \
    --from-file=server-ca.pem \
    --from-file=server-cert.pem \
    --from-file=server-key.pem \
    --from-file=client-cert.pem \
    --from-file=client-key.pem \
    --from-file=sql-config.cnf \
    --from-file=sql-config.json \
    --from-file=db-root-password
