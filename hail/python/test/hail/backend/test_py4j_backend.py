import types
from typing import Callable

import orjson
import py4j.clientserver
import pytest
import requests
from py4j.java_gateway import GatewayConnection

from hail.backend.backend import ActionTag
from hail.backend.py4j_backend import Py4JBackend, _log_suppress_exceptions, _make_interrupt_safe
from hail.utils.java import FatalError

pytestmark = pytest.mark.uninitialized


class Interrupt(BaseException):
    # Like pytest-timeout's Failed or KeyboardInterrupt: not an Exception,
    # so it bypasses py4j's error handling in send_command.
    pass


def raising(exception):
    def raise_(*ignored, **also_ignored):
        raise exception

    return raise_


interrupt = raising(Interrupt)


class FakeJBackend:
    def __init__(self, cancel: Callable[[], None] = lambda: None):
        self._cancel = cancel

    def pyCancel(self):
        self._cancel()


@pytest.fixture
def rpc(monkeypatch):
    def go(post, jbackend):
        # `_rpc` only touches the http server's port and `pyCancel`, so a
        # stand-in object suffices; no gateway or JVM is involved
        backend = types.SimpleNamespace(
            _jhttp_server=types.SimpleNamespace(port=lambda: 5555),
            _jbackend=jbackend,
        )
        monkeypatch.setattr(requests, 'post', post)
        return Py4JBackend._rpc(backend, ActionTag.EXECUTE, {})

    return go


def response(status_code, content):
    return types.SimpleNamespace(status_code=status_code, content=content)


def java_error(status_code=500):
    return response(status_code, orjson.dumps({'short': 'short', 'expanded': 'expanded', 'error_id': -1}))


def fake_connection(handler):
    # a fresh class per call: _make_interrupt_safe patches the class itself
    class FakeConnection:
        def __init__(self):
            self.closed_with_reset = None

        def send_command(self, command):
            return handler(command)

        def close(self, reset=False):
            self.closed_with_reset = reset

    _make_interrupt_safe(FakeConnection)
    return FakeConnection()


def test_interrupt_propagates():
    connection = fake_connection(interrupt)
    with pytest.raises(Interrupt):
        connection.send_command('command')


def test_interrupt_closes_the_connection():
    connection = fake_connection(interrupt)
    with pytest.raises(Interrupt):
        connection.send_command('command')
    assert connection.closed_with_reset is True


def test_exceptions_do_not_close_the_connection():
    connection = fake_connection(raising(ValueError))
    with pytest.raises(ValueError):
        connection.send_command('command')
    assert connection.closed_with_reset is None


def test_successful_commands_pass_through():
    connection = fake_connection(lambda command: f'!yv{command}')
    assert connection.send_command('command') == '!yvcommand'


def test_patching_is_idempotent():
    cls = type(fake_connection(lambda command: command))
    patched = cls.send_command
    _make_interrupt_safe(cls)
    assert cls.send_command is patched


@pytest.mark.parametrize('cls', [GatewayConnection, py4j.clientserver.ClientServerConnection])
def test_py4j_connections_are_interrupt_safe(cls):
    assert getattr(cls.send_command, '_hail_interrupt_safe', False)


def test_log_suppress_shutdown_exceptions_swallows_exceptions():
    _log_suppress_exceptions(raising(ValueError))


def test_log_suppress_shutdown_exceptions_propagates_interrupts():
    with pytest.raises(Interrupt):
        _log_suppress_exceptions(interrupt)


@pytest.mark.parametrize('failure', [Interrupt, requests.exceptions.ConnectionError])
def test_rpc_failures_cancel_the_operation_and_propagate(rpc, failure):
    cancelled = []
    with pytest.raises(failure):
        rpc(raising(failure), FakeJBackend(cancel=lambda: cancelled.append(True)))
    assert cancelled


def test_rpc_cancel_failures_do_not_mask_the_original_failure(rpc):
    with pytest.raises(Interrupt):
        rpc(interrupt, FakeJBackend(cancel=raising(ValueError)))


def test_rpc_returns_successful_responses(rpc):
    assert (
        rpc(
            lambda *a, **k: response(200, b'payload'),
            FakeJBackend(cancel=lambda: pytest.fail("cancel should not have been called")),
        )
        == b'payload'
    )


def test_rpc_java_errors_raise_fatal_errors(rpc):
    with pytest.raises(FatalError):
        rpc(
            lambda *a, **k: java_error(),
            FakeJBackend(cancel=lambda: pytest.fail("cancel should not have been called")),
        )
