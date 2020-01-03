import pytest
from ..helpers import *

import hail.experimental.time as htime

@skip_unless_spark_backend()
def test_strftime():
    assert hl.eval(htime.strftime("%A, %B %e, %Y. %r", 876541523, "America/New_York")) == "Friday, October 10, 1997. 11:45:23 PM"
