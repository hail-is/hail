from batch.server.server import run_once
import requests
import urllib3
from urllib3 import HTTPConnectionPool
import socket
import unittest
import logging
import sys

logging.basicConfig(stream=sys.stderr)


class ServerTest(unittest.TestCase):
    def test_errors(self):
        def urllib3_timeout(err):
            raise urllib3.exceptions.ReadTimeoutError(HTTPConnectionPool(
                host='127.0.0.1', port=5000), 'url://stuff', err)

        def standard_timeout(err):
            raise requests.exceptions.ReadTimeout(err)

        def socket_timeout(err):
            raise socket.timeout(err)
        print("RUNNING")

        try:
            run_once(urllib3_timeout, "urllib3 timeout")
            run_once(standard_timeout, "standard timeouut")
            run_once(socket_timeout, "socket timeout")
        except Exception:
            self.fail("Couldn't handle timeout exceptions")


if __name__ == "__main__":
    unittest.main()
