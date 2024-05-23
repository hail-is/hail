from typing import Dict, Any
import traceback


def exception_to_dict(exc: Exception) -> Dict[str, Any]:
    return {
        'class': type(exc).__name__,
        'args': exc.args,
        'traceback': traceback.format_exception(type(exc), exc, exc.__traceback__),
    }
