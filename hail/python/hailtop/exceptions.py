class MultipleExceptions(Exception):
    def __init__(self, message, causes):
        self.message = message
        self.causes = causes

    def __str__(self):
        return f'{self.message} caused by {[str(m) for m in self.causes]}'
