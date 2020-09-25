from typing import Dict, NamedTuple, Optional, Any
import json
import os


class SQLConfig(NamedTuple):
    host: str
    port: int
    user: str
    password: str
    instance: str
    connection_name: str
    db: Optional[str]
    ssl_ca: str
    ssl_cert: str
    ssl_key: str
    ssl_mode: str

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d = {'host': self.host,
             'port': self.port,
             'user': self.user,
             'password': self.password,
             'instance': self.instance,
             'connection_name': self.connection_name,
             'ssl-ca': self.ssl_ca,
             'ssl-cert': self.ssl_cert,
             'ssl-key': self.ssl_key,
             'ssl-mode': self.ssl_mode}
        if self.db is not None:
            d['db'] = self.db
        return d

    def to_cnf(self) -> str:
        host_user_password = f'''[client]
host={self.host}
user={self.user}
password="{self.password}"
'''
        database_setting = f'database={self.db}\n' if self.db is not None else ''
        ssl_settings = f'''ssl-ca={self.ssl_ca}
ssl-cert={self.ssl_cert}
ssl-key={self.ssl_key}
ssl-mode={self.ssl_mode}
'''
        return host_user_password + database_setting + ssl_settings

    def check(self):
        assert self.host is not None
        assert self.port is not None
        assert self.user is not None
        assert self.password is not None
        assert self.instance is not None
        assert self.connection_name is not None
        assert self.ssl_ca is not None
        assert self.ssl_cert is not None
        assert self.ssl_key is not None
        if not os.path.isfile(self.ssl_cert):
            raise ValueError(f'specified ssl-cert, {self.ssl_cert}, does not exist')
        if not os.path.isfile(self.ssl_key):
            raise ValueError(f'specified ssl-key, {self.ssl_key}, does not exist')
        if not os.path.isfile(self.ssl_ca):
            raise ValueError(f'specified ssl-ca, {self.ssl_ca}, does not exist')

    @staticmethod
    def from_json(s: str) -> 'SQLConfig':
        return SQLConfig.from_dict(json.loads(s))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'SQLConfig':
        for k in ('host', 'port', 'user', 'password',
                  'instance', 'connection_name',
                  'ssl-ca', 'ssl-cert', 'ssl-key', 'ssl-mode'):
            assert k in d, f'{k} should be in {d}'
            assert d[k] is not None, f'{k} should not be None in {d}'
        return SQLConfig(host=d['host'],
                         port=d['port'],
                         user=d['user'],
                         password=d['password'],
                         instance=d['instance'],
                         connection_name=d['connection_name'],
                         db=d.get('db'),
                         ssl_ca=d['ssl-ca'],
                         ssl_cert=d['ssl-cert'],
                         ssl_key=d['ssl-key'],
                         ssl_mode=d['ssl-mode'])


def create_secret_data_from_config(config: SQLConfig,
                                   server_ca: str,
                                   client_cert: str,
                                   client_key: str
                                   ) -> Dict[str, str]:
    secret_data = dict()
    secret_data['sql-config.json'] = config.to_json()
    secret_data['sql-config.cnf'] = config.to_cnf()
    secret_data['server-ca.pem'] = server_ca
    secret_data['client-cert.pem'] = client_cert
    secret_data['client-key.pem'] = client_key
    return secret_data
