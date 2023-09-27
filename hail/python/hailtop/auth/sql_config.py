from typing import Dict, NamedTuple, Optional, Any
import json
import os


class SQLConfig(NamedTuple):
    host: str
    port: int
    user: str
    password: str
    instance: Optional[str]
    connection_name: Optional[str]
    db: Optional[str]
    ssl_ca: Optional[str]
    ssl_cert: Optional[str]
    ssl_key: Optional[str]
    ssl_mode: Optional[str]

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
             'ssl-mode': self.ssl_mode}
        if self.db is not None:
            d['db'] = self.db
        if self.using_mtls():
            d['ssl-cert'] = self.ssl_cert
            d['ssl-key'] = self.ssl_key
        return d

    def to_cnf(self) -> str:
        cnf = f'''[client]
host={self.host}
user={self.user}
port={self.port}
password="{self.password}"
ssl-ca={self.ssl_ca}
ssl-mode={self.ssl_mode}
'''
        if self.db is not None:
            cnf += f'database={self.db}\n'
        if self.using_mtls():
            cnf += f'ssl-cert={self.ssl_cert}\n'
            cnf += f'ssl-key={self.ssl_key}\n'
        return cnf

    def check(self):
        assert self.host is not None
        assert self.port is not None
        assert self.user is not None
        assert self.password is not None
        assert self.instance is not None
        assert self.connection_name is not None
        if self.ssl_cert is not None:
            assert self.ssl_key is not None
            if not os.path.isfile(self.ssl_cert):
                raise ValueError(f'specified ssl-cert, {self.ssl_cert}, does not exist')
            if not os.path.isfile(self.ssl_key):
                raise ValueError(f'specified ssl-key, {self.ssl_key}, does not exist')
        else:
            assert self.ssl_key is None
        assert self.ssl_ca is not None
        if not os.path.isfile(self.ssl_ca):
            raise ValueError(f'specified ssl-ca, {self.ssl_ca}, does not exist')

    def using_mtls(self) -> bool:
        if self.ssl_cert is not None:
            assert self.ssl_key is not None
            return True
        assert self.ssl_key is None
        return False

    @staticmethod
    def from_json(s: str) -> 'SQLConfig':
        return SQLConfig.from_dict(json.loads(s))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'SQLConfig':
        for k in ('host', 'port', 'user', 'password',
                  'instance', 'connection_name',
                  'ssl-ca', 'ssl-mode'):
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
                         ssl_cert=d.get('ssl-cert'),
                         ssl_key=d.get('ssl-key'),
                         ssl_mode=d['ssl-mode'])

    @staticmethod
    def local_insecure_config() -> 'SQLConfig':
        return SQLConfig(
            host='localhost',
            port=3306,
            user='root',
            password='pw',
            db=os.environ.get('HAIL_SQL_DATABASE'),
            instance=None,
            connection_name=None,
            ssl_ca=None,
            ssl_cert=None,
            ssl_key=None,
            ssl_mode=None,
        )


def create_secret_data_from_config(config: SQLConfig,
                                   server_ca: str,
                                   client_cert: Optional[str],
                                   client_key: Optional[str]
                                   ) -> Dict[str, str]:
    secret_data = {}
    secret_data['sql-config.json'] = config.to_json()
    secret_data['sql-config.cnf'] = config.to_cnf()
    secret_data['server-ca.pem'] = server_ca
    if client_cert is not None:
        assert client_key is not None
        secret_data['client-cert.pem'] = client_cert
        secret_data['client-key.pem'] = client_key
    else:
        assert client_cert is None and client_key is None
    return secret_data
