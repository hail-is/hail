import uuid
import hailjwt as hj


def test_round_trip():
    c = hj.JWTClient(uuid.uuid4().hex)
    json = {'hello': 'world'}
    assert c.jwt_decode(c.jwt_encode(json)) == json


def test_get_domain():
    assert hj.get_domain('notebook.hail.is') == 'hail.is'
