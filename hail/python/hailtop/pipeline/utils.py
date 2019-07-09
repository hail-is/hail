
class PipelineException(Exception):
    def __init__(self, msg=''):
        self.msg = msg
        super(PipelineException, self).__init__(msg)
