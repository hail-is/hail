import pytest

from auth.auth_utils import is_valid_username, validate_credentials_secret_name_input
from auth.exceptions import AuthUserError


@pytest.mark.parametrize(
    "input_secret_name",
    [
        'abc',
        'abc.def',
        'abc-def',
        'abc.def-ghi',
        'abc-def.ghi',
        'a3a',
        'a3a.3a',
        'a3a-3a',
        'a3a.3a-3a',
        'a3a-3a.3a',
        'a3a.3a.3a',
        'a',
        'a.a',
        'a-a',
        'a.a-a',
        'a-a.a',
        'a.a.a',
        'a3',
        'a3.a3',
        'a3-a3',
        'a3.a3-a3',
        'a3-a3.a3',
        'a3.a3.a3',
        '3',
        '3.3',
        '3-3',
        '3.3-3',
        '3-3.3',
        '3.3.3',
    ],
)
def test_validate_credentials_secret_name_input_valid(input_secret_name):
    validate_credentials_secret_name_input(input_secret_name)


@pytest.mark.parametrize(
    "input_secret_name",
    [
        'abc!',
        'abc_def',
        'abc!def',
        'abc.def.',
        'abc.-def',
        'abc.-def.ghi',
        'abc.def-.ghi',
        '.abc',
        'abc.',
        'abc..def',
        'abc--def',
        'abc.def--ghi',
    ],
)
def test_validate_credentials_secret_name_input_invalid(input_secret_name):
    with pytest.raises(AuthUserError):
        validate_credentials_secret_name_input(input_secret_name)


@pytest.mark.parametrize("input_username", ['abc', 'abc-def', 'a3a', 'a3a-3a', '123'])
def test_is_valid_username_valid(input_username):
    assert is_valid_username(input_username)


@pytest.mark.parametrize(
    "input_username",
    [
        None,
        '',
        'ABC',
        'abc!',
        'a3a.3a',
        'abc_def',
        'abc!def',
        'abc.def.',
        'abc.def',
        'abc.def-ghi',
        'abc-def.ghi',
        'abc--def',
        'abc.def--ghi',
        'abc.def-.ghi',
        '.abc',
        'abc.',
        'abc..def',
    ],
)
def test_is_valid_username_invalid(input_username):
    assert not is_valid_username(input_username)
