from aiohttp import web


class AuthUserError(Exception):
    def __init__(self, message, severity):
        super().__init__(message)
        self.message = message
        self.ui_error_type = severity

    def http_response(self) -> web.HTTPError:
        return web.HTTPBadRequest(reason=self.message)


class EmptyLoginID(AuthUserError):
    def __init__(self, username):
        super().__init__(f"Login id for user '{username}' must be a non-empty string.", 'error')


class DuplicateLoginID(AuthUserError):
    def __init__(self, username, login_id):
        super().__init__(f"Login id '{login_id}' already exists for user '{username}'.", 'error')


class DuplicateUsername(AuthUserError):
    def __init__(self, username, login_id):
        super().__init__(f"Username '{username}' already exists with login id '{login_id}'.", 'error')


class InvalidUsername(AuthUserError):
    def __init__(self, username):
        super().__init__(f"Invalid username '{username}'. Must be a non-empty alphanumeric string.", 'error')


class InvalidType(AuthUserError):
    def __init__(self, field_name, input, expected_type):
        super().__init__(f"Expected '{field_name}' is of type {expected_type}. Found type {type(input)}", 'error')


class MultipleUserTypes(AuthUserError):
    def __init__(self, username):
        super().__init__(f"User '{username}' cannot be both a developer and a service account.", 'error')


class MultipleExistingUsers(AuthUserError):
    def __init__(self, username, login_id):
        super().__init__(
            f"Multiple users with user name '{username}' and login id '{login_id}' appear in the database.",
            'error',
        )

    def http_response(self):
        return web.HTTPInternalServerError(reason=self.message)


class UnknownUser(AuthUserError):
    def __init__(self, username):
        super().__init__(f"Unknown user '{username}'.", 'error')

    def http_response(self):
        return web.HTTPNotFound(reason=self.message)


class PreviouslyDeletedUser(AuthUserError):
    def __init__(self, username):
        super().__init__(f"User '{username}' has already been deleted.", 'error')
