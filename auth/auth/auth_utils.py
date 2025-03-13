import re
from typing import Optional

from .exceptions import AuthUserError


def validate_credentials_secret_name_input(secret_name: Optional[str]):
    if secret_name is None:
        return

    regex = re.compile(r'^[a-z0-9]([.\-]?[a-z0-9])*[a-z0-9]?$')
    if not regex.match(secret_name):
        raise AuthUserError(
            f'invalid credentials_secret_name {secret_name}. Must match RFC1123 (lowercase alphanumeric plus "." and "-", start and end with alphanumeric)',
            'error',
        )
