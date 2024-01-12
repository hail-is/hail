import datetime

from hailtop import timex


def test_google_cloud_storage_example():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404Z')
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo=datetime.timezone.utc)
    assert actual == expected


def test_timezone_positive():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404+04:00')

    tzinfo = datetime.timezone(datetime.timedelta(hours=4))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_timezone_negative():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404-04:00')

    tzinfo = datetime.timezone(datetime.timedelta(hours=-4))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_timezone_positive_with_minutes():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404+04:35')

    tzinfo = datetime.timezone(datetime.timedelta(hours=4, minutes=35))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_timezone_negative_with_minutes():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404-04:35')

    tzinfo = datetime.timezone(datetime.timedelta(hours=-4, minutes=-35))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_space_instead_of_T():
    actual = timex.parse_rfc3339('2022-12-27 16:48:06Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 0, tzinfo)
    assert actual == expected

    actual = timex.parse_rfc3339('2022-12-27 16:48:06.404Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected

    actual = timex.parse_rfc3339('2022-12-27 16:48:06.404-04:35')

    tzinfo = datetime.timezone(datetime.timedelta(hours=-4, minutes=-35))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_lowercase_T():
    actual = timex.parse_rfc3339('2022-12-27t16:48:06Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 0, tzinfo)
    assert actual == expected

    actual = timex.parse_rfc3339('2022-12-27t16:48:06.404Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected

    actual = timex.parse_rfc3339('2022-12-27t16:48:06.404-04:35')

    tzinfo = datetime.timezone(datetime.timedelta(hours=-4, minutes=-35))
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_lowercase_z():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 0, tzinfo)
    assert actual == expected

    actual = timex.parse_rfc3339('2022-12-27T16:48:06.404z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 404000, tzinfo)
    assert actual == expected


def test_one_fractional_second_digit():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.1Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 100000, tzinfo)
    assert actual == expected


def test_six_fractional_second_digits():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.123456Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 123456, tzinfo)
    assert actual == expected


def test_ten_fractional_second_digits_should_round_correctly_1():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.1234567890Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 123457, tzinfo)
    assert actual == expected


def test_ten_fractional_second_digits_should_round_correctly_2():
    actual = timex.parse_rfc3339('2022-12-27T16:48:06.1234564890Z')

    tzinfo = datetime.timezone.utc
    expected = datetime.datetime(2022, 12, 27, 16, 48, 6, 123456, tzinfo)
    assert actual == expected
