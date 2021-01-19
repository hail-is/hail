import hail as hl
from test.hail.helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

UNBLOCKED_UNBUFFERED_SPEC = '{"name":"StreamBufferSpec"}'
BLOCKED_UNBUFFERED_SPEC = '''{"name":"BlockingBufferSpec","blockSize":65536,
                              "child":{"name":"StreamBlockBufferSpec"}}'''


def assert_round_trip(exp, codec):
    (pt, b) = hl.experimental.encode(exp, codec=codec)
    result = hl.experimental.decode(exp.dtype, pt, b, codec=codec)
    assert hl.eval(exp) == result


def assert_round_trip_all_specs(exp):
    assert_round_trip(exp, UNBLOCKED_UNBUFFERED_SPEC)
    assert_round_trip(exp, BLOCKED_UNBUFFERED_SPEC)


@skip_unless_spark_backend()
def test_encode_basics():
    (_, b) = hl.experimental.encode(hl.literal(1), codec='{"name":"StreamBufferSpec"}')
    assert b.hex() == '01000000'

    (_, b) = hl.experimental.encode(hl.literal(-1), codec='{"name":"StreamBufferSpec"}')
    assert b.hex() == 'ffffffff'


@skip_unless_spark_backend()
def test_decode_basics():
    result = hl.experimental.decode(hl.tint32,
                                    'PInt32',
                                    bytes.fromhex('01000000'),
                                    UNBLOCKED_UNBUFFERED_SPEC)
    assert result == 1

    result = hl.experimental.decode(hl.tint32,
                                    'PInt32',
                                    bytes.fromhex('ffffffff'),
                                    UNBLOCKED_UNBUFFERED_SPEC)
    assert result == -1


@skip_unless_spark_backend()
def test_round_trip_basics():
    assert_round_trip_all_specs(hl.literal(1))


@skip_unless_spark_backend()
def test_complex_round_trips():
    assert_round_trip_all_specs(hl.struct())
    assert_round_trip_all_specs(hl.empty_array(hl.tint32))
    assert_round_trip_all_specs(hl.empty_set(hl.tint32))
    assert_round_trip_all_specs(hl.empty_dict(hl.tint32, hl.tint32))
    assert_round_trip_all_specs(hl.locus('1', 100))
    assert_round_trip_all_specs(hl.struct(x=3))
    assert_round_trip_all_specs(hl.set([3, 4, 5, 3]))
    assert_round_trip_all_specs(hl.array([3, 4, 5]))
    assert_round_trip_all_specs(hl.dict({3: 'a', 4: 'b', 5: 'c'}))

    assert_round_trip_all_specs(hl.struct(x=hl.dict({3: 'a', 4: 'b', 5: 'c'}),
                                          y=hl.array([3, 4, 5]),
                                          z=hl.set([3, 4, 5, 3])))
