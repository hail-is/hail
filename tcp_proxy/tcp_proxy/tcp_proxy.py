import os
import ssl
import signal
import asyncio
from contextlib import closing
import aiohttp
import logging
from hailtop.tcp import BYTE_ORDER, STRING_ENCODING
from .constants import BUFFER_SIZE
from hailtop.config import get_deploy_config
from hailtop.tls import get_in_cluster_server_ssl_context
from hailtop.tcp import (open_direct_connection, open_proxied_connection,
                         HailTCPConnectionError)
from hailtop.auth import async_get_userinfo
from hailtop.auth import session_id_encode_to_str


log = logging.getLogger('scorecard')
HAIL_DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']


async def pipe(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        connection_lost: asyncio.Event,
        name: str,
        log_info
):
    while not reader.at_eof() and not connection_lost.is_set():
        try:
            incoming_data = await asyncio.wait_for(reader.read(BUFFER_SIZE), timeout=10)
        except asyncio.TimeoutError:
            continue
        except ConnectionResetError:
            log_info(f'{name}: read-side dropped the connection')
            connection_lost.set()
            return

        writer.write(incoming_data)

        try:
            await writer.drain()
        except ConnectionResetError:
            log_info(f'{name} write-side dropped the connection')
            connection_lost.set()
            return

    if reader.at_eof():
        log_info(f'{name}: EOF')
    else:
        assert connection_lost.is_set()
        log.info(f'{name}: other side of pipe was lost')

    if writer.can_write_eof():
        writer.write_eof()
    connection_lost.set()
    return


SERVER_TLS_CONTEXT = get_in_cluster_server_ssl_context()


async def handle(source_reader: asyncio.StreamReader, source_writer: asyncio.StreamWriter):
    source_addr = source_writer.get_extra_info('peername')

    session_id_bytes = await source_reader.readexactly(32)
    session_id = session_id_encode_to_str(session_id_bytes)
    internal_session_id_bytes = await source_reader.readexactly(32)
    internal_session_id = session_id_encode_to_str(internal_session_id_bytes)
    namespace_name_length = int.from_bytes(await source_reader.readexactly(4), byteorder=BYTE_ORDER, signed=False)
    namespace_name = (await source_reader.readexactly(namespace_name_length)).decode(encoding=STRING_ENCODING)
    service_name_length = int.from_bytes(await source_reader.readexactly(4), byteorder=BYTE_ORDER, signed=False)
    service_name = (await source_reader.readexactly(service_name_length)).decode(encoding=STRING_ENCODING)
    port = int.from_bytes(await source_reader.readexactly(2), byteorder=BYTE_ORDER, signed=False)

    try:
        hostname = f'{service_name}.{namespace_name}'
        target_addr = (hostname, port)

        userinfo = await async_get_userinfo(session_id=session_id)
        if userinfo is None:
            raise HailTCPConnectionError(f'invalid credentials {session_id}')

        if '.' in service_name or '.' in namespace_name:
            raise HailTCPConnectionError('malformed request')

        if namespace_name == HAIL_DEFAULT_NAMESPACE:
            log.info(f'attempting direct connection {target_addr} -> {source_addr}')
            connection_id, target_reader, target_writer = await open_direct_connection(
                service_name, namespace_name, port,
                session_ids=(session_id_bytes, internal_session_id_bytes))
            log.info(f'sucessful direct connection {target_addr} -> {source_addr}')
        elif HAIL_DEFAULT_NAMESPACE == 'default':  # default router may forward to namespaced ones
            if userinfo['is_developer'] != 1:
                raise HailTCPConnectionError('not developer')

            # we do not verify namespaced certs
            client_tls_context = ssl.create_default_context()
            client_tls_context.check_hostname = False
            client_tls_context.verify_mode = ssl.CERT_NONE

            internal_deploy_config = get_deploy_config().with_service('auth', namespace_name)
            internal_userinfo = await async_get_userinfo(deploy_config=internal_deploy_config,
                                                         session_id=internal_session_id,
                                                         client_session=aiohttp.ClientSession(
                                                             raise_for_status=True,
                                                             timeout=aiohttp.ClientTimeout(total=5),
                                                             connector=aiohttp.TCPConnector(
                                                                 ssl=client_tls_context)))
            if internal_userinfo is None:
                raise HailTCPConnectionError(f'invalid internal credentials {internal_session_id}')

            log.info(f'attempting proxied connection {target_addr} -> {source_addr}')
            connection_id, target_reader, target_writer = await open_proxied_connection(
                proxy_hostname=f'tcp-proxy.{namespace_name}',
                proxy_port=5000,
                service=service_name,
                ns=namespace_name,
                port=port,
                session_ids=(internal_session_id_bytes, b'\x00' * 32),
                ssl=client_tls_context)
            log.info(f'successful proxied connection {target_addr} -> {source_addr}')
        else:
            raise HailTCPConnectionError(
                f'cannot route to namespace from {HAIL_DEFAULT_NAMESPACE}')
    except HailTCPConnectionError:
        log.info(f'failed to connect {target_addr} <- {source_addr}', exc_info=True)
        source_writer.write(b'\x00')
        source_writer.close()
        return
    except Exception:
        log.info(f'unexpected connection failure {target_addr} <- {source_addr}', exc_info=True)
        source_writer.write(b'\x00')
        source_writer.close()
        return

    connection_metadata = {
        'connection_id': connection_id,
        'source_addr': source_addr,
        'target_addr': target_addr
    }

    def log_info(*args, **kwargs):
        kwargs['extra'] = connection_metadata
        log.info(*args, **kwargs)

    log_info('OPEN')

    source_writer.write(b'\x01')
    assert len(connection_id.bytes) == 16
    source_writer.write(connection_id.bytes)

    source_to_target = None
    target_to_source = None
    try:
        connection_lost = asyncio.Event()
        source_to_target = asyncio.create_task(
            pipe(source_reader, target_writer, connection_lost, 'source->target', log_info))
        target_to_source = asyncio.create_task(
            pipe(target_reader, source_writer, connection_lost, 'target->source', log_info))
        await source_to_target
        await target_to_source
    finally:
        log_info('CLOSING')
        target_writer.close()
        source_writer.close()
        if source_to_target is not None:
            source_to_target.cancel()
        if target_to_source is not None:
            target_to_source.cancel()
        log_info('CLOSED')


async def run_server(host, port):
    with closing(await asyncio.start_server(
            handle,
            host,
            port,
            loop=asyncio.get_event_loop(),
            ssl=SERVER_TLS_CONTEXT)):
        print(f'Serving on {host}:{port}')
        while True:
            await asyncio.sleep(3600)


async def shutdown(signal, loop):
    log.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not
             asyncio.current_task()]

    for task in tasks:
        task.cancel()

    logging.info("Cancelling outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def run():
    host = '0.0.0.0'
    port = 5000
    with closing(asyncio.get_event_loop()) as loop:
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

        loop.run_until_complete(run_server(host, port))
