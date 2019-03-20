import jwt
import re
import secrets

import hailjwt as hj


def test_round_trip():
    c = hj.JWTClient(secrets.token_bytes(256))
    json = {'hello': 'world'}
    assert c.decode(c.encode(json)) == json


def test_fewer_than_256_bits_is_error():
    try:
        hj.JWTClient(secrets.token_bytes(255))
        assert False
    except ValueError as err:
        assert re.search('found secret key with 255 bytes', str(err))


def test_bad_input_is_error():
    try:
        c = hj.JWTClient(secrets.token_bytes(256))
        c.decode('garbage')
        assert False
    except jwt.exceptions.DecodeError:
        pass


def test_get_domain():
    assert hj.get_domain('notebook.hail.is') == 'hail.is'
