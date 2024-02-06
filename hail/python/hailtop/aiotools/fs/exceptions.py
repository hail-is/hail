class UnexpectedEOFError(Exception):
    pass


class FileAndDirectoryError(Exception):
    pass


class IsABucketError(FileNotFoundError):
    pass
