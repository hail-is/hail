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


def is_valid_username(username: str) -> bool:
    """Check if a username is valid.

    Requirements:
    1. Only alphanumeric characters and hyphens allowed
    2. Hyphens cannot be at start or end
    3. Hyphens cannot be adjacent to each other

    Args:
        username: The username to validate

    Returns:
        bool: True if username meets all requirements, False otherwise
    """
    if not username:  # Check for empty string
        return False

    # Check for hyphens at start or end
    if username.startswith('-') or username.endswith('-'):
        return False

    # Check for adjacent hyphens
    if '--' in username:
        return False

    # Check that all characters are numeric or lowercase or hyphen:
    return all(c.isascii() and (c.isdigit() or c.islower() or c == '-') for c in username)
