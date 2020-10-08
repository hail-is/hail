from aiohttp import web

class BatchUserError(Exception):
    def __init__(self, message, severity):
        self.message = message
        self.ui_error_type = severity

    def api_response(self):
        return web.HTTPForbidden(reason=self.message)


class NonExistentBillingProjectError(Exception):
    def __init__(self, billing_project):
        super().__init__(f'Billing project {billing_project} does not exist.', 'error')

    def api_response(self):
        return web.HTTPNotFound(reason=self.message)


class DeletedBillingProjectError(Exception):
    def __init__(self, billing_project):
        super().__init__(f'Billing project {billing_project} is deleted and can no longer be modified.', 'error')