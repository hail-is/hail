from aiohttp import web


class BatchUserError(Exception):
    def __init__(self, message, severity):
        super().__init__(message)
        self.message = message
        self.ui_error_type = severity

    def http_response(self):
        return web.HTTPForbidden(reason=self.message)


class NonExistentBillingProjectError(BatchUserError):
    def __init__(self, billing_project):
        super().__init__(f'Billing project {billing_project} does not exist.', 'error')

    def http_response(self):
        return web.HTTPNotFound(reason=self.message)


class ClosedBillingProjectError(BatchUserError):
    def __init__(self, billing_project):
        super().__init__(f'Billing project {billing_project} is closed and cannot be modified.', 'error')


class InvalidBillingLimitError(BatchUserError):
    def __init__(self, billing_limit):
        super().__init__(f'Invalid billing_limit {billing_limit}.', 'error')

    def http_response(self):
        return web.HTTPBadRequest(reason=self.message)


class NonExistentBatchError(BatchUserError):
    def __init__(self, batch_id):
        super().__init__(f'Batch {batch_id} does not exist.', 'error')


class NonExistentJobGroupError(BatchUserError):
    def __init__(self, batch_id, job_group_id):
        super().__init__(f'Job Group {(batch_id, job_group_id)} does not exist.', 'error')


class JobGroupAlreadyExistsError(BatchUserError):
    def __init__(self, batch_id, job_group_path):
        super().__init__(f'Job Group {(batch_id, job_group_path)} already exists.', 'error')


class NonExistentJobGroupPathError(BatchUserError):
    def __init__(self, batch_id, job_group_path):
        super().__init__(f'Job Group {(batch_id, job_group_path)} does not exist.', 'error')


class NonExistentJobGroupIDError(BatchUserError):
    def __init__(self, batch_id, job_group_id):
        super().__init__(f'Job Group {(batch_id, job_group_id)} does not exist.', 'error')


class InvalidJobGroupPathError(BatchUserError):
    def __init__(self, job_group_path):
        super().__init__(f'Job group path "{job_group_path}" is invalid. Must start with "/"', 'error')

    def http_response(self):
        return web.HTTPBadRequest(reason=self.message)


class OpenBatchError(BatchUserError):
    def __init__(self, batch_id):
        super().__init__(f'Batch {batch_id} is open.', 'error')


class BatchOperationAlreadyCompletedError(Exception):
    def __init__(self, message, severity):
        super().__init__(message)
        self.message = message
        self.ui_error_type = severity


class QueryError(BatchUserError):
    def __init__(self, message):
        super().__init__(message, 'error')
        self.message = message

    def http_response(self):
        return web.HTTPBadRequest(reason=self.message)
