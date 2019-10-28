class DatabaseCallError(Exception):
    def __init__(self, out):
        super().__init__(out)
        self.out = out
