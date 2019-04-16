import jwt
import re
import secrets

import hailjwt as hj


def test_round_trip():
    c = hj.JWTClient(hj.JWTClient.generate_key())
    json = {'hello': 'world'}
    assert c.decode(c.encode(json)) == json


def test_fewer_than_256_bits_is_error():
    try:
        hj.JWTClient(secrets.token_bytes(31))
        assert False
    except ValueError as err:
        assert re.search('found secret key with 31 bytes', str(err))


def test_bad_input_is_error():
    try:
        c = hj.JWTClient(hj.JWTClient.generate_key())
        c.decode('garbage')
        assert False
    except jwt.exceptions.DecodeError:
        pass


def test_decode_message_from_different_key_is_error():
    try:
        c = hj.JWTClient(hj.JWTClient.generate_key())
        attacker = hj.JWTClient(hj.JWTClient.generate_key())
        json = {'hello': 'world'}
        c.decode(attacker.encode(json))
        assert False
    except jwt.exceptions.DecodeError:
        pass


def test_modified_cyphertext_is_error():
    try:
        c1 = hj.JWTClient(hj.JWTClient.generate_key())
        c2 = hj.JWTClient(hj.JWTClient.generate_key())
        json = {'hello': 'world'}
        c2.decode(str(c1.encode(json)) + "evil")
        assert False
    except jwt.exceptions.DecodeError:
        pass


def test_get_domain():
    assert hj.get_domain('notebook.hail.is') == 'hail.is'


def test_unsafe_decode():
    c = hj.JWTClient(hj.JWTClient.generate_key())
    json = {'hello': 'world'}
    assert c.unsafe_decode(c.encode(json)) == json
