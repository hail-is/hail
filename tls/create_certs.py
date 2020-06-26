import argparse
import json
import yaml
import shutil
import subprocess as sp
import tempfile
import os

parser = argparse.ArgumentParser(prog='create_certs.py',
                                 description='create hail certs')
parser.add_argument('namespace', type=str, help='kubernetes namespace')
parser.add_argument('config_file', type=str, help='YAML format config file')
parser.add_argument('root_key_file', type=str, help='the root key file')
parser.add_argument('root_cert_file', type=str, help='the root cert file')
parser.add_argument('--create-k8s-secrets', dest="create_k8s_secrets", help='do not create the k8s secrets', action='store_true')
parser.add_argument('--no-create-k8s-secrets', dest="create_k8s_secrets", help='do not create the k8s secrets', action='store_false')
parser.add_argument('--root-path', type=str, help='root path for secret files; a python format string with one variable in scope: principal')
parser.set_defaults(create_k8s_secrets=True, root_path='/ssl-config')

args = parser.parse_args()
namespace = args.namespace
with open(args.config_file) as f:
    arg_config = yaml.safe_load(f)
root_key_file = os.path.abspath(args.root_key_file)
root_cert_file = os.path.abspath(args.root_cert_file)
create_k8s_secrets = args.create_k8s_secrets
_root_path = args.root_path.rstrip('/')


def root_path(principal):
    return _root_path.format(principal=principal)


class WorkingDirectoryContext:
    def __init__(self, path):
        self.path = path
        self.oldpath = None

    def __enter__(self):
        self.oldpath = os.getcwd()
        os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)

    def __exit__(self, exc_type, value, traceback):
        os.chdir(self.oldpath)


def cd(path):
    return WorkingDirectoryContext(path)


def echo_check_call(cmd):
    print(" ".join(cmd))
    sp.check_call(cmd)


def create_key_and_cert(p):
    domain = p['domain']
    unmanaged = p.get('unmanaged', False)
    if unmanaged and namespace != 'default':
        return
    key_file = f'key.pem'
    csr_file = f'csr.csr'
    cert_file = f'cert.pem'
    key_store_file = f'key-store.p12'
    names = [
        domain,
        f'{domain}.{namespace}',
        f'{domain}.{namespace}.svc.cluster.local'
    ]

    echo_check_call([
        'openssl', 'genrsa',
        '-out', key_file,
        '4096'
    ])
    echo_check_call([
        'openssl', 'req',
        '-new',
        '-subj', f'/CN={names[0]}',
        '-key', key_file,
        '-out', csr_file
    ])
    extfile = tempfile.NamedTemporaryFile(mode='w', delete=False)
    # this whole extfile nonsense is because OpenSSL has known, unfixed bugs
    # in the x509 command. These really ought to be in the CSR.
    # https://www.openssl.org/docs/man1.1.0/man1/x509.html#BUGS
    # https://security.stackexchange.com/questions/150078/missing-x509-extensions-with-an-openssl-generated-certificate
    extfile.write(f'subjectAltName = {",".join("DNS:" + n for n in names)}\n')
    extfile.close()
    echo_check_call(['cat', extfile.name])
    echo_check_call([
        'openssl', 'x509',
        '-req',
        '-in', csr_file,
        '-CA', root_cert_file,
        '-CAkey', root_key_file,
        '-extfile', extfile.name,
        '-CAcreateserial',
        '-out', cert_file,
        '-days', '365'
    ])
    echo_check_call([
        'openssl',
        'pkcs12',
        '-export',
        '-inkey', key_file,
        '-in', cert_file,
        '-name', f'key-store',
        '-out', key_store_file,
        '-passout', 'pass:dummypw'
    ])
    return {'key': key_file, 'cert': cert_file, 'key_store': key_store_file}


def create_trust(trust_type):  # pylint: disable=unused-argument
    trust_file = f'{trust_type}.pem'
    trust_store_file = f'{trust_type}-store.jks'
    with open(trust_file, 'w') as out:
        # FIXME: mTLS, only trust certain principals
        with open(root_cert_file, 'r') as root_cert:
            shutil.copyfileobj(root_cert, out)
    echo_check_call([
        'keytool',
        '-noprompt',
        '-import',
        '-alias', f'{trust_type}-cert',
        '-file', trust_file,
        '-keystore', trust_store_file,
        '-storepass', 'dummypw'
    ])
    return {'trust': trust_file, 'trust_store': trust_store_file}


def create_json_config(root, incoming_trust, outgoing_trust, key, cert, key_store):
    principal_config = {
        'outgoing_trust': f'{root}/{outgoing_trust["trust"]}',
        'outgoing_trust_store': f'{root}/{outgoing_trust["trust_store"]}',
        'incoming_trust': f'{root}/{incoming_trust["trust"]}',
        'incoming_trust_store': f'{root}/{incoming_trust["trust_store"]}',
        'key': f'{root}/{key}',
        'cert': f'{root}/{cert}',
        'key_store': f'{root}/{key_store}'
    }
    config_file = 'ssl-config.json'
    with open(config_file, 'w') as out:
        out.write(json.dumps(principal_config))
        return [config_file]


def create_nginx_config(root, incoming_trust, outgoing_trust, key, cert):
    http_config_file = 'ssl-config-http.conf'
    proxy_config_file = 'ssl-config-proxy.conf'
    with open(proxy_config_file, 'w') as proxy, open(http_config_file, 'w') as http:
        proxy.write(f'proxy_ssl_certificate         {root}/{cert};\n')
        proxy.write(f'proxy_ssl_certificate_key     {root}/{key};\n')
        proxy.write(f'proxy_ssl_trusted_certificate {root}/{outgoing_trust["trust"]};\n')
        proxy.write('proxy_ssl_verify              on;\n')
        proxy.write('proxy_ssl_verify_depth        1;\n')
        proxy.write('proxy_ssl_session_reuse       on;\n')

        http.write(f'ssl_certificate {root}/{cert};\n')
        http.write(f'ssl_certificate_key {root}/{key};\n')
        http.write(f'ssl_client_certificate {root}/{incoming_trust["trust"]};\n')
        http.write('ssl_verify_client optional;\n')
        # FIXME: mTLS
        # http.write('ssl_verify_client on;\n')
    return [http_config_file, proxy_config_file]


def create_curl_config(root, incoming_trust, outgoing_trust, key, cert):
    config_file = 'ssl-config.curlrc'
    with open(config_file, 'w') as out:
        out.write(f'key       {root}/{key}\n')
        out.write(f'cert      {root}/{cert}\n')
        out.write(f'cacert    {root}/{outgoing_trust["trust"]}\n')
    return [config_file]


def create_config(root, incoming_trust, outgoing_trust, key, cert, key_store, kind):
    if kind == 'json':
        return create_json_config(root, incoming_trust, outgoing_trust, key, cert, key_store)
    if kind == 'curl':
        return create_curl_config(root, incoming_trust, outgoing_trust, key, cert)
    assert kind == 'nginx'
    return create_nginx_config(root, incoming_trust, outgoing_trust, key, cert)


def create_principal(principal, domain, kind, key, cert, key_store, unmanaged):
    if unmanaged and namespace != 'default':
        return
    root = root_path(principal)
    incoming_trust = create_trust('incoming')
    outgoing_trust = create_trust('outgoing')
    configs = create_config(root, incoming_trust, outgoing_trust, key, cert, key_store, kind)
    if not create_k8s_secrets:
        return
    with tempfile.NamedTemporaryFile() as k8s_secret:
        sp.check_call(
            ['kubectl', 'create', 'secret', 'generic', f'ssl-config-{principal}',
             f'--namespace={namespace}',
             f'--from-file={key}',
             f'--from-file={cert}',
             f'--from-file={key_store}',
             f'--from-file={incoming_trust["trust"]}',
             f'--from-file={incoming_trust["trust_store"]}',
             f'--from-file={outgoing_trust["trust"]}',
             f'--from-file={outgoing_trust["trust_store"]}',
             *[f'--from-file={c}' for c in configs],
             '--dry-run', '-o', 'yaml'],
            stdout=k8s_secret)
        sp.check_call(['kubectl', 'apply', '-f', k8s_secret.name])


assert 'principals' in arg_config, arg_config

principal_by_name = dict()
for p in arg_config['principals']:
    name = p['name']
    with cd(name):
        principal_by_name[name] = {**p, **create_key_and_cert(p)}
for name, p in principal_by_name.items():
    with cd(name):
        create_principal(name,
                         p['domain'],
                         p['kind'],
                         p['key'],
                         p['cert'],
                         p['key_store'],
                         p.get('unmanaged', False))
