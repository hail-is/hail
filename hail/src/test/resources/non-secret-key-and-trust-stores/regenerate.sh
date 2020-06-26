python3 -c '
import yaml
import shutil
conf = yaml.safe_load(open("config.yaml"))
for p in conf["principals"]:
    shutil.rmtree(p["name"], ignore_errors=True)
'

openssl req -x509 \
        -newkey rsa:4096 \
        -keyout root-key.pem \
        -out root-cert.pem \
        -days 365 \
        -subj '/CN=localhost' \
        -nodes

python3 ../../../../../tls/create_certs.py \
        default \
        config.yaml \
        root-key.pem \
        root-cert.pem \
        --no-create-k8s-secrets \
        --root-path='src/test/resources/non-secret-key-and-trust-stores/{principal}'

rm -rf root-cert.pem root-cert.srl root-key.pem  # not necessary after principal creation
