import hail as hl
from test.hail.helpers import *

UNBLOCKED_UNBUFFERED_SPEC = '{"name":"StreamBufferSpec"}'


def assert_round_trip(exp, codec=UNBLOCKED_UNBUFFERED_SPEC):
    b = hl.experimental.encode(exp, codec=codec)
    result = hl.experimental.decode(exp.dtype, b)
    assert hl.eval(exp) == result


@run_in('spark')
def test_encode_basics():
    b = hl.experimental.encode(hl.literal(1), codec='{"name":"StreamBufferSpec"}')
    assert b.hex() == '01000000'

    b = hl.experimental.encode(hl.literal(-1), codec='{"name":"StreamBufferSpec"}')
    assert b.hex() == 'ffffffff'


@run_in('spark')
def test_decode_basics():
    result = hl.experimental.decode(hl.tint32,
                                    bytes.fromhex('01000000'))
    assert result == 1

    result = hl.experimental.decode(hl.tint32,
                                    bytes.fromhex('ffffffff'))
    assert result == -1


@run_in('spark')
def test_round_trip_basics():
    assert_round_trip(hl.literal(1))


@run_in('spark')
def test_complex_round_trips():
    assert_round_trip(hl.struct())
    assert_round_trip(hl.empty_array(hl.tint32))
    assert_round_trip(hl.empty_set(hl.tint32))
    assert_round_trip(hl.empty_dict(hl.tint32, hl.tint32))
    assert_round_trip(hl.locus('1', 100))
    assert_round_trip(hl.struct(x=3))
    assert_round_trip(hl.set([3, 4, 5, 3]))
    assert_round_trip(hl.array([3, 4, 5]))
    assert_round_trip(hl.dict({3: 'a', 4: 'b', 5: 'c'}))
    assert_round_trip(hl.struct(x=hl.dict({3: 'a', 4: 'b', 5: 'c'}),
                                y=hl.array([3, 4, 5]),
                                z=hl.set([3, 4, 5, 3])))
