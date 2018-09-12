Mastering the API
=================

Logging
~~~~~~~

Writing output of :meth:`.Table.show` to log file
.................................................

:**tags**: logging, show, print

:**description**: Change the handling of strings in the :meth:`.Table.show` method
                  by defining a custom handler.

:**code**:

        >>> ht.show(handler=lambda x: logging.info(x))  # doctest: +SKIP

:**dependencies**: :func:`.Table.show`, :meth:`.Expression.show`