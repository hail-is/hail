import asyncio
import errno
import io
import logging
import os
import shutil
import struct
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import cancel_and_retrieve_all_exceptions, check_shell_output, sleep_before_try, time_msecs, time_ns

log = logging.getLogger('resource_usage')


iptables_lock = asyncio.Lock()


class ResourceUsageMonitor:
    VERSION = 2
    missing_value = None

    @staticmethod
    def no_data() -> bytes:
        return ResourceUsageMonitor.version_to_bytes()

    @staticmethod
    def version_to_bytes() -> bytes:
        return struct.pack('>q', ResourceUsageMonitor.VERSION)

    @staticmethod
    def decode_to_df(data: bytes) -> Optional[pd.DataFrame]:
        try:
            return ResourceUsageMonitor._decode_to_df(data)
        except Exception:
            log.exception('corrupt resource usage file found', stack_info=True)
            return None

    @staticmethod
    def _decode_to_df(data: bytes) -> Optional[pd.DataFrame]:
        if len(data) == 0:
            return None

        (version,) = struct.unpack_from('>q', data, 0)
        assert 1 <= version <= ResourceUsageMonitor.VERSION, version

        dtype = [
            ('time_msecs', '>i8'),
            ('memory_in_bytes', '>i8'),
            ('cpu_usage', '>f8'),
        ]

        if version > 1:
            assert version == ResourceUsageMonitor.VERSION, version
            dtype += [
                ('non_io_storage_in_bytes', '>i8'),
                ('io_storage_in_bytes', '>i8'),
                ('network_bandwidth_upload_in_bytes_per_second', '>f8'),
                ('network_bandwidth_download_in_bytes_per_second', '>f8'),
            ]
        np_array = np.frombuffer(data, offset=8, dtype=dtype)

        return pd.DataFrame.from_records(np_array)

    def __init__(
        self,
        container_name: str,
        container_overlay: str,
        io_volume_mount: Optional[str],
        veth_host: str,
        output_file_path: str,
        fs: AsyncFS,
    ):
        assert veth_host is not None

        self.container_name = container_name
        self.container_overlay = container_overlay
        self.io_volume_mount = io_volume_mount
        self.veth_host = veth_host
        self.output_file_path = output_file_path
        self.fs = fs

        self.is_attached_disk = io_volume_mount is not None and os.path.ismount(io_volume_mount)

        self.last_time_ns: Optional[int] = None
        self.last_cpu_ns: Optional[int] = None

        self.last_download_bytes: Optional[int] = None
        self.last_upload_bytes: Optional[int] = None
        self.last_time_msecs: Optional[int] = None

        self.out: Optional[io.BufferedWriter] = None

        self.task: Optional[asyncio.Task] = None

    def write_header(self):
        assert self.out
        data = self.version_to_bytes()
        self.out.write(data)
        self.out.flush()

    def cpu_ns(self) -> Optional[int]:
        # See below for a nice breakdown of the cpu cgroupv2:
        # https://facebookmicrosites.github.io/cgroup2/docs/cpu-controller.html#interface-files
        #
        # and here for the authoritative source:
        # https://git.kernel.org/pub/scm/linux/kernel/git/tj/cgroup.git/tree/Documentation/admin-guide/cgroup-v2.rst#n1038
        usage_file = f'/sys/fs/cgroup/{self.container_name}/cpu.stat'
        if os.path.exists(usage_file):
            with open(usage_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    stat, val = line.strip().split(' ')
                    if stat == 'usage_usec':
                        return int(val) * 1000
        return None

    def percent_cpu_usage(self) -> Optional[float]:
        now_time_ns = time_ns()
        now_cpu_ns = self.cpu_ns()

        if now_cpu_ns is None or self.last_cpu_ns is None or self.last_time_ns is None:
            cpu_usage = None
        else:
            cpu_usage = (now_cpu_ns - self.last_cpu_ns) / (now_time_ns - self.last_time_ns)

        self.last_time_ns = now_time_ns
        self.last_cpu_ns = now_cpu_ns
        return cpu_usage

    def memory_usage_bytes(self) -> Optional[int]:
        # See below for a nice breakdown of the memory cgroupv2:
        # https://facebookmicrosites.github.io/cgroup2/docs/memory-controller.html#core-interface-files
        #
        # and here for the authoritative source:
        # https://git.kernel.org/pub/scm/linux/kernel/git/tj/cgroup.git/tree/Documentation/admin-guide/cgroup-v2.rst#n1156
        usage_file = f'/sys/fs/cgroup/{self.container_name}/memory.current'
        try:
            if os.path.exists(usage_file):
                with open(usage_file, 'r', encoding='utf-8') as f:
                    return int(f.read().rstrip())
        except OSError as e:
            # OSError: [Errno 19] No such device
            if e.errno == 19:
                return None
            raise
        return None

    def overlay_storage_usage_bytes(self) -> int:
        return shutil.disk_usage(self.container_overlay).used

    def io_storage_usage_bytes(self) -> int:
        if self.io_volume_mount is not None:
            return shutil.disk_usage(self.io_volume_mount).used
        return 0

    async def network_bandwidth(self) -> Tuple[Optional[float], Optional[float]]:
        async with iptables_lock:
            now_time_msecs = time_msecs()

            iptables_output, stderr = await check_shell_output(
                f'''
iptables -t mangle -L -v -n -x -w | grep "{self.veth_host}" | awk '{{ if ($6 == "{self.veth_host}" || $7 == "{self.veth_host}") print $2, $6, $7 }}'
'''
            )
        if stderr:
            log.exception(stderr)
            return (None, None)

        output = iptables_output.decode('utf-8').rstrip().splitlines()
        assert len(output) == 2, str((output, self.veth_host))

        now_upload_bytes = None
        now_download_bytes = None
        for line in output:
            fields = line.split()
            bytes_transmitted = int(fields[0])

            if fields[1] == self.veth_host and fields[2] != self.veth_host:
                now_upload_bytes = bytes_transmitted
            else:
                assert fields[1] != self.veth_host and fields[2] == self.veth_host, line
                now_download_bytes = bytes_transmitted

        assert now_upload_bytes is not None and now_download_bytes is not None, output

        if self.last_upload_bytes is None or self.last_download_bytes is None or self.last_time_msecs is None:
            self.last_time_msecs = time_msecs()
            self.last_upload_bytes = now_upload_bytes
            self.last_download_bytes = now_download_bytes
            return (None, None)

        upload_bandwidth = (now_upload_bytes - self.last_upload_bytes) / (now_time_msecs - self.last_time_msecs)
        download_bandwidth = (now_download_bytes - self.last_download_bytes) / (now_time_msecs - self.last_time_msecs)

        upload_bandwidth_mb_sec = (upload_bandwidth / 1024 / 1024) * 1000
        download_bandwidth_mb_sec = (download_bandwidth / 1024 / 1024) * 1000

        self.last_time_msecs = now_time_msecs
        self.last_upload_bytes = now_upload_bytes
        self.last_download_bytes = now_download_bytes

        return (upload_bandwidth_mb_sec, download_bandwidth_mb_sec)

    async def measure(self):
        now = time_msecs()
        memory_usage_bytes = self.memory_usage_bytes()
        percent_cpu_usage = self.percent_cpu_usage()

        if memory_usage_bytes is None or percent_cpu_usage is None:
            return

        overlay_usage_bytes = self.overlay_storage_usage_bytes()
        io_usage_bytes = self.io_storage_usage_bytes()
        non_io_usage_bytes = overlay_usage_bytes if self.is_attached_disk else overlay_usage_bytes - io_usage_bytes
        network_upload_bytes_per_second, network_download_bytes_per_second = await self.network_bandwidth()

        if network_upload_bytes_per_second is None or network_download_bytes_per_second is None:
            return

        data = struct.pack(
            '>2qd2q2d',
            now,
            memory_usage_bytes,
            percent_cpu_usage,
            non_io_usage_bytes,
            io_usage_bytes,
            network_upload_bytes_per_second,
            network_download_bytes_per_second,
        )

        assert self.out
        self.out.write(data)
        self.out.flush()

    async def read(self):
        if os.path.exists(self.output_file_path):
            return await self.fs.read(self.output_file_path)
        return ResourceUsageMonitor.no_data()

    async def __aenter__(self):
        async def periodically_measure():
            start = time_msecs()
            cancelled = False
            tries = 0
            while True:
                try:
                    await self.measure()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                except OSError as err:
                    if err.errno == errno.ENOSPC:
                        cancelled = True
                        raise
                    log.exception(f'while monitoring {self.container_name}')
                except Exception:
                    log.exception(f'while monitoring {self.container_name}')
                finally:
                    if not cancelled:
                        tries += 1
                        if time_msecs() - start < 5_000:
                            await asyncio.sleep(0.1)
                        else:
                            await sleep_before_try(tries, max_delay_ms=5_000)

        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        self.out = open(self.output_file_path, 'wb')  # pylint: disable=consider-using-with
        self.write_header()

        self.task = asyncio.create_task(periodically_measure())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task is not None:
            await cancel_and_retrieve_all_exceptions([self.task])
            self.task = None

        if self.out is not None:
            self.out.close()
            self.out = None
