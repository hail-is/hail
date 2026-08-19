import py4j.clientserver
import pytest
from py4j.java_gateway import GatewayConnection

from hail.backend.py4j_backend import _log_suppress_shutdown_exceptions, _make_interrupt_safe

pytestmark = pytest.mark.uninitialized


class Interrupt(BaseException):
    # Like pytest-timeout's Failed or KeyboardInterrupt: not an Exception,
    # so it bypasses py4j's error handling in send_command.
    pass


def interrupt(*ignored):
    raise Interrupt


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
    def raise_value_error(command):
        raise ValueError(command)

    connection = fake_connection(raise_value_error)
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
    def raise_value_error():
        raise ValueError

    _log_suppress_shutdown_exceptions(raise_value_error)


def test_log_suppress_shutdown_exceptions_propagates_interrupts():
    with pytest.raises(Interrupt):
        _log_suppress_shutdown_exceptions(interrupt)
