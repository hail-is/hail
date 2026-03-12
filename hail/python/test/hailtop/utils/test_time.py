from hailtop.utils.time import parse_timestamp_msecs, time_msecs, time_msecs_str


def test_time_msecs_roundtrip():
    time = time_msecs()
    assert time == parse_timestamp_msecs(time_msecs_str(time))
