import json


def create_secret_data_from_config(config, server_ca, client_cert, client_key):
    assert all(config.get(x) is not None
               for x in ('host', 'user', 'password', 'db', 'ssl-ca', 'ssl-cert',
                         'ssl-key', 'ssl-mode')), config.keys()
    secret_data = dict()
    secret_data['sql-config.json'] = json.dumps(config)
    secret_data['sql-config.cnf'] = f'''[client]
host={config["host"]}
user={config["user"]}
password="{config["password"]}"
database={config["db"]}
ssl-ca={config["ssl-ca"]}
ssl-cert={config["ssl-cert"]}
ssl-key={config["ssl-key"]}
ssl-mode={config["ssl-mode"]}
'''
    secret_data['server-ca.pem'] = server_ca
    secret_data['client-cert.pem'] = client_cert
    secret_data['client-key.pem'] = client_key
    return secret_data
