from typing import Tuple, Optional, Dict, Any
import uuid
import asyncio
import logging
from .auth import session_id_decode_from_str, get_tokens
from .config import get_deploy_config
from .tls import get_context_specific_client_ssl_context


BYTE_ORDER = 'big'
STRING_ENCODING = 'utf-8'
log = logging.getLogger('tcp')


# Router TCP Authentication Protocol
#
# IN: 32 bytes, session_id, bytes
# IN: 32 bytes, internal_session_id, bytes, may be all zeros
# IN: 4 bytes, namespace_name length, unsigned integer
# IN: ?? bytes, namespace_name, UTF-8 string
# IN: 4 bytes, service_name length, unsigned integer
# IN: ?? bytes, service_name, UTF-8 string
# IN: 2 bytes, port, unsigned integer
# OUT: 1 byte, connect_is_successful, unsigned integer, 1 = success, 0 = not-success
# OUT: 16 bytes, connection_id, bytes

# Service TCP Authentication Protocol
#
# IN: 32 bytes, session_id, bytes
# IN: 32 bytes, internal_session_id, bytes, may be all zeros
# OUT: 1 byte, connect_is_successful, unsigned integer, 1 = success, 0 = not-success
# OUT: 16 bytes, connection_id, bytes


class HailTCPConnectionError(Exception):
    pass


CLIENT_TLS_CONTEXT = get_context_specific_client_ssl_context()


async def open_connection(service: str,
                          port: int,
                          **kwargs
                          ) -> Tuple[uuid.UUID, asyncio.StreamReader, asyncio.StreamWriter]:
    assert port < (1 << 16)
    deploy_config = get_deploy_config()
    ns = deploy_config.service_ns(service)
    location = deploy_config.location()
    tokens = get_tokens()

    if location == 'k8s':
        kwargs['session_ids'] = (session_id_decode_from_str(tokens.namespace_token_or_error(ns)),
                                 b'\x00' * 32)
        return await open_direct_connection(service, ns, port, **kwargs)

    if location == 'gce':
        proxy_hostname = 'hail'
    else:
        assert location == 'external'
        proxy_hostname = 'hail.is'

    kwargs['session_ids'] = (session_id_decode_from_str(tokens.namespace_token_or_error('default')),
                             session_id_decode_from_str(tokens.namespace_token_or_error(ns)))
    return await open_proxied_connection(proxy_hostname, 5000,
                                         service, ns, port,
                                         **kwargs)


async def open_direct_connection(service: str,
                                 ns: str,
                                 port: int,
                                 **kwargs
                                 ) -> Tuple[uuid.UUID, asyncio.StreamReader, asyncio.StreamWriter]:
    return await _open_connection(f'{service}.{ns}', port, None, **kwargs)


async def open_proxied_connection(proxy_hostname: str,
                                  proxy_port: int,
                                  service: str,
                                  ns: str,
                                  port: int,
                                  **kwargs
                                  ) -> Tuple[uuid.UUID, asyncio.StreamReader, asyncio.StreamWriter]:
    return await _open_connection(proxy_hostname,
                                  proxy_port,
                                  proxy_to={'service': service, 'ns': ns, 'port': port},
                                  **kwargs)


async def _open_connection(hostname: str,
                           port: int,
                           proxy_to: Optional[Dict[str, Any]],
                           session_ids: Tuple[bytes, bytes],
                           **kwargs
                           ) -> Tuple[uuid.UUID, asyncio.StreamReader, asyncio.StreamWriter]:
    if 'ssl' not in kwargs:
        kwargs['ssl'] = CLIENT_TLS_CONTEXT
    if 'loop' not in kwargs:
        kwargs['loop'] = asyncio.get_event_loop()

    while True:
        reader, writer = await asyncio.open_connection(
            hostname,
            port,
            **kwargs)

        await write_session_ids(writer, session_ids)

        if proxy_to is not None:
            ns = proxy_to['ns']
            service = proxy_to['service']
            port = proxy_to['port']
            log.info('establishing proxy connection')
            writer.write(len(ns).to_bytes(4, BYTE_ORDER))
            writer.write(ns.encode(STRING_ENCODING))
            writer.write(len(service).to_bytes(4, BYTE_ORDER))
            writer.write(service.encode(STRING_ENCODING))
            writer.write((port).to_bytes(2, BYTE_ORDER))

        await writer.drain()
        try:
            is_success = await reader.readexactly(1)
        except asyncio.IncompleteReadError:
            log.info('end of file encountered before reading anything, retrying connection')
            continue
        if is_success != b'\x01':
            raise HailTCPConnectionError(f'{hostname}:{port} {is_success!r}')
        break

    connection_id_bytes = await reader.readexactly(16)
    connection_id = uuid.UUID(bytes=connection_id_bytes)

    return connection_id, reader, writer


async def write_session_ids(writer: asyncio.StreamWriter, session_ids: Tuple[bytes, bytes]):
    assert len(session_ids) == 2, session_ids
    default_session_id, namespace_session_id = session_ids

    assert len(default_session_id) == 32
    assert len(namespace_session_id) == 32

    writer.write(default_session_id)
    writer.write(namespace_session_id)
