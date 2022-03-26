import asyncio
import os
import struct
from typing import Optional

import numpy as np
import pandas as pd

from hailtop.utils import periodically_call, retry_long_running, time_msecs, time_ns


class ResourceUsageMonitor:
    VERSION = 1

    @staticmethod
    def no_data() -> bytes:
        return ResourceUsageMonitor.version_to_bytes()

    @staticmethod
    def version_to_bytes() -> bytes:
        return struct.pack('>q', ResourceUsageMonitor.VERSION)

    @staticmethod
    def decode_to_df(data: bytes) -> Optional[pd.DataFrame]:
        if len(data) == 0:
            return None

        (version,) = struct.unpack_from('>q', data, 0)
        assert version == ResourceUsageMonitor.VERSION, version

        dtype = [('time_msecs', '>i8'), ('memory_in_bytes', '>i8'), ('cpu_usage', '>f8')]
        np_array = np.frombuffer(data, offset=8, dtype=dtype)
        return pd.DataFrame.from_records(np_array)

    def __init__(self, container_name: str, output_file_path: str):
        self.container_name = container_name
        self.output_file_path = output_file_path

        self.last_time_ns: Optional[int] = None
        self.last_cpu_ns: Optional[int] = None

        self.out = open(output_file_path, 'wb')  # pylint: disable=consider-using-with
        self.write_header()

        self.task: Optional[asyncio.Future] = None

    def write_header(self):
        data = ResourceUsageMonitor.version_to_bytes()
        self.out.write(data)
        self.out.flush()

    def cpu_ns(self) -> Optional[int]:
        usage_file = f'/sys/fs/cgroup/cpu/{self.container_name}/cpuacct.usage'
        if os.path.exists(usage_file):
            with open(usage_file, 'r', encoding='utf-8') as f:
                return int(f.read().rstrip())
        return None

    def percent_cpu_usage(self) -> Optional[float]:
        now_time_ns = time_ns()
        now_cpu_ns = self.cpu_ns()

        if now_cpu_ns is None or self.last_cpu_ns is None or self.last_time_ns is None:
            self.last_time_ns = now_time_ns
            self.last_cpu_ns = now_cpu_ns
            return None

        cpu_usage = (now_cpu_ns - self.last_cpu_ns) / (now_time_ns - self.last_time_ns)

        self.last_time_ns = now_time_ns
        self.last_cpu_ns = now_cpu_ns

        return cpu_usage

    def memory_usage_bytes(self) -> Optional[int]:
        usage_file = f'/sys/fs/cgroup/memory/{self.container_name}/memory.usage_in_bytes'
        if os.path.exists(usage_file):
            with open(usage_file, 'r', encoding='utf-8') as f:
                return int(f.read().rstrip())
        return None

    async def measure(self):
        now = time_msecs()
        memory_usage_bytes = self.memory_usage_bytes()
        percent_cpu_usage = self.percent_cpu_usage()

        if memory_usage_bytes is None or percent_cpu_usage is None:
            return

        data = struct.pack('>2qd', now, memory_usage_bytes, percent_cpu_usage)
        self.out.write(data)
        self.out.flush()

    async def __aenter__(self):
        self.task = asyncio.ensure_future(
            retry_long_running(f'monitor {self.container_name} resource usage', periodically_call, 5, self.measure)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task is not None:
            self.task.cancel()
        self.out.close()
