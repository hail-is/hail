import base64
import sys
import json
import yaml
import shutil
import subprocess as sp
import tempfile

namespace = sys.argv[1]
assert namespace is not None, namespace
arg_config = yaml.safe_load(sys.argv[2])


def create_cert_and_key(principal, domain):
    key_file = f'{principal}-key.pem'
    cert_file = f'{principal}-cert.pem'
    names = [
        domain,
        f'{domain}.{namespace}',
        f'{domain}.{namespace}.svc.cluster.local'
    ]
    sp.check_call([
        'openssl', 'req',
        '-new',
        '-x509',
        '-subj', f'/CN={names[0]}',
        '-addext', f'subjectAltName = {" ".join("DNS:" + n for n in names)}',
        '-nodes',  # no password, key itself is cleartext
        '-newkey', 'rsa:4096',
        '-keyout', key_file,
        '-out', cert_file
    ])
    return (key_file, cert_file)


def create_trust(principal, trusted_principals):
    trust_file = f'{principal}-trust.pem'
    with open(trust_file, 'w') as out:
        for p in trusted_principals:
            with open(f'{p}-cert.pem', 'r') as cert:
                shutil.copyfileobj(cert, out)
            with open(f'previous-{p}-cert.pem', 'r') as previous_cert:
                shutil.copyfileobj(previous_cert, out)
    return trust_file


def create_json_config(principal, trust, cert, key):
    principal_config = {
        'ssl-mode': arg_config['ssl-mode'],
        'ssl-ca': f'/ssl-config/{trust}',
        'ssl-cert': f'/ssl-config/{cert}',
        'ssl-key': f'/ssl-config/{key}'
    }
    config_file = f'ssl-config.json'
    with open(config_file, 'w') as out:
        out.write(json.dumps(principal_config))
        return [config_file]


def create_nginx_config(principal, trust, cert, key):
    http_config_file = f'ssl-config-http.conf'
    proxy_config_file = f'ssl-config-proxy.conf'
    with open(proxy_config_file, 'w') as proxy, open(http_config_file, 'w') as http:
        if arg_config['ssl-mode'] == 'VERIFY_CA':
            proxy.write('proxy_ssl_certificate         /ssl-config/{cert};\n')
            proxy.write('proxy_ssl_certificate_key     /ssl-config/{key};\n')
            proxy.write('proxy_ssl_trusted_certificate /ssl-config/{trust};\n')
            proxy.write('proxy_ssl_verify              on;\n')
            proxy.write('proxy_ssl_verify_depth        1;\n')
            proxy.write('proxy_ssl_session_reuse       on;\n')

            http.write(f'ssl_certificate /ssl-config/{cert};\n')
            http.write(f'ssl_certificate_key /ssl-config/{key};\n')
            http.write(f'ssl_trusted_certificate /ssl-config/{trust};\n')
            http.write(f'ssl_verify_client on;\n')
        elif arg_config['ssl-mode'] == 'REQUIRED':
            proxy.write(f'proxy_ssl_certificate         /ssl-config/{cert};\n')
            proxy.write(f'proxy_ssl_certificate_key     /ssl-config/{key};\n')
            proxy.write(f'proxy_ssl_trusted_certificate /ssl-config/{trust};\n')
            proxy.write(f'proxy_ssl_session_reuse       on;\n')

            http.write(f'ssl_certificate /ssl-config/{cert};\n')
            http.write(f'ssl_certificate_key /ssl-config/{key};\n')
            http.write(f'ssl_trusted_certificate /ssl-config/{trust};\n')
        elif arg_config['ssl-mode'] == 'DISABLED':
            proxy.write('')
            http.write('')
        else:
            assert False, 'only DISABLED, REQURIED, and VERIFY_CA are ' \
                'supported for ssl-mode. {arg_config.get("ssl-mode")}'
        return [http_config_file, proxy_config_file]


def create_curl_config(principal, trust, cert, key):
    config_file = f'ssl-config.curlrc'
    with open(config_file, 'w') as out:
        if arg_config['ssl-mode'] == 'VERIFY_CA':
            out.write(f'cert      /ssl-config/{cert}\n')
            out.write(f'key       /ssl-config/{key}\n')
            out.write(f'cacert    /ssl-config/{trust}\n')
        elif arg_config['ssl-mode'] == 'REQUIRED':
            out.write(f'cert      /ssl-config/{cert}\n')
            out.write(f'key       /ssl-config/{key}\n')
            out.write(f'cacert    /ssl-config/{trust}\n')
            out.write(f'insecure')
        elif arg_config['ssl-mode'] == 'DISABLED':
            pass
        else:
            assert False, 'only DISABLED, REQURIED, and VERIFY_CA are ' \
                'supported for ssl-mode. {arg_config.get("ssl-mode")}'
        return [config_file]


def create_config(principal, trust, cert, key, kind):
    if kind == 'json':
        return create_json_config(principal, trust, cert, key)
    if kind == 'curl':
        return create_curl_config(principal, trust, cert, key)
    assert kind == 'nginx'
    return create_nginx_config(principal, trust, cert, key)


def create_principal(principal, trusted_principals, domain, kind, key, cert):
    trust = create_trust(principal, trusted_principals)
    configs = create_config(principal, trust, cert, key, kind)
    with tempfile.NamedTemporaryFile() as k8s_secret:
        sp.check_call(
            ['kubectl', 'create', 'secret', 'generic', f'ssl-config-{principal}',
             f'--namespace={namespace}',
             f'--from-file={key}',
             f'--from-file={cert}',
             f'--from-file={trust}',
             *[f'--from-file={c}' for c in configs],
             '--dry-run', '-o', 'yaml'],
            stdout=k8s_secret)
        sp.check_call(['kubectl', 'apply', '-f', k8s_secret.name])


def download_previous_certs():
    for p in arg_config['principals']:
        principal = p["name"]
        result = sp.run(
            ['kubectl', 'get', 'secret', f'ssl-config-{principal}',
             f'--namespace={namespace}', '-o', 'json'],
            stderr=sp.PIPE,
            stdout=sp.PIPE)
        if result.returncode == 1:
            if f'Error from server (NotFound)' in result.stderr.decode():
                cert = b''
            else:
                raise ValueError(f'something went wrong: {result.stderr.decode()}\n---\n{result.stdout.decode()}')
        else:
            secret = json.loads(result.stdout.decode())
            cert = base64.standard_b64decode(secret['data'][f'{principal}-cert.pem'])
        with open(f'previous-{principal}-cert.pem', 'wb') as f:
            f.write(cert)


assert 'principals' in arg_config, arg_config
assert 'ssl-mode' in arg_config, arg_config

key_and_cert = {
    p['name']: create_cert_and_key(p['name'], p['domain'])
    for p in arg_config['principals']
}

download_previous_certs()

for p in arg_config['principals']:
    key, cert = key_and_cert[p['name']]
    create_principal(p['name'], p.get('trust', []), p['domain'], p['kind'], key, cert)
