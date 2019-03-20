import jwt
import re
import secrets

import hailjwt as hj


def test_round_trip():
    c = hj.JWTClient(secrets.randbits(256))
    json = {'hello': 'world'}
    assert c.decode(c.encode(json)) == json


def test_fewer_than_256_bits_is_error():
    try:
        hj.JWTClient(secrets.randbits(255 * 8))
        assert False
    except ValueError as err:
        assert re.search('found secret key with 255 bytes', err.message)


def test_bad_input_is_error():
    try:
        c = hj.JWTClient(secrets.randbits(256 * 8))
        c.decode('garbage')
        assert False
    except jwt.exceptions.DecodeError:
        pass


def test_get_domain():
    assert hj.get_domain('notebook.hail.is') == 'hail.is'
