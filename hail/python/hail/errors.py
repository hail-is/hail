from typing import Union


class HailUserError(Exception):
    """:class:`.HailUserError` is an error thrown by Hail when the user makes an error."""


class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""

    def __init__(self, msg, error_id: int = -1):
        super().__init__(msg)
        self._error_id = error_id

    def maybe_user_error(self, ir) -> Union['FatalError', HailUserError]:
        error_sources = ir.base_search(lambda x: x._error_id == self._error_id)
        if len(error_sources) == 0:
            return self

        better_stack_trace = error_sources[0]._stack_trace
        error_message = str(self)
        message_and_trace = (f'{error_message}\n'
                             '------------\n'
                             'Hail stack trace:\n'
                             f'{better_stack_trace}')
        return HailUserError(message_and_trace)
