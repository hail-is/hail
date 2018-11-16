class UserError(Exception):
    def __init__(self, data, status_code=400):
        assert 400 <= status_code < 500
        Exception.__init__(self, str(data))
        self.data = data
        self.status_code = status_code
