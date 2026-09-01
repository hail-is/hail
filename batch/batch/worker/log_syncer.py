import asyncio
import logging
import os
import signal
from contextlib import suppress

import async_timeout

log = logging.getLogger(__name__)

_active_log_syncers: set['LogSyncer'] = set()

LOG_SYNC_SCRIPT = '/usr/local/bin/log-sync.sh'
LOG_SYNC_STATE_DIR = '/run/batch-worker/log-sync'
LOG_SYNC_TARGET_BYTES_PER_S = 100 * 1024  # target upload rate; sleep = file_size / this, clamped to [min, max]
LOG_SYNC_MIN_INTERVAL = 15  # seconds
LOG_SYNC_MAX_INTERVAL = 600  # seconds
LOG_SYNC_FINISH_TIMEOUT = 120


class LogSyncer:
    """Manages a bash subprocess that incrementally syncs a job log file to GCS.

    The instruction file in LOG_SYNC_STATE_DIR is the sole coordination channel between
    worker.py and the subprocess. It is deliberately kept outside the job scratch space
    so the user cannot interfere with it.
    """

    def __init__(
        self,
        log_path: str,
        remote_url: str,
        instruction_file: str,
        proc: 'asyncio.subprocess.Process',
    ):
        self._log_path = log_path
        self._remote_url = remote_url
        self._instruction_file = instruction_file
        self._log_copier_proc = proc

    @classmethod
    async def start(cls, log_path: str, remote_url: str, batch_id: int, job_id: int, attempt_id: str) -> 'LogSyncer':
        instruction_file = os.path.join(LOG_SYNC_STATE_DIR, f'{batch_id}_{job_id}_{attempt_id}.conf')
        cls._write_instruction_file(instruction_file, log_path, remote_url, 'running')
        # Touch the log file so GCS gets an empty object on the first sync cycle rather than a 404.
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8'):
            pass
        proc = await asyncio.create_subprocess_exec(
            '/bin/bash',
            LOG_SYNC_SCRIPT,
            instruction_file,
            stdout=None,  # inherit worker stdout/stderr so all output appears in container logs
            stderr=None,
        )
        try:
            async with async_timeout.timeout(30):
                while proc.returncode is None:
                    try:
                        with open(instruction_file, encoding='utf-8') as f:
                            if 'trap_installed=1\n' in f.read():
                                break
                    except FileNotFoundError:
                        pass
                    await asyncio.sleep(0.01)
                else:
                    raise RuntimeError(f'log syncer exited before becoming ready (code {proc.returncode})')
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            with suppress(FileNotFoundError):
                os.unlink(instruction_file)
            raise RuntimeError('log syncer did not become ready within 30s') from exc
        log.info(f'started log syncer pid={proc.pid} {log_path} -> {remote_url}')
        syncer = cls(log_path, remote_url, instruction_file, proc)
        _active_log_syncers.add(syncer)
        return syncer

    @staticmethod
    def _write_instruction_file(path: str, log_path: str, remote_url: str, state: str) -> None:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(f'log={log_path}\n')
            f.write(f'remote={remote_url}\n')
            f.write(f'state={state}\n')
            f.write(f'target_bytes_per_s={LOG_SYNC_TARGET_BYTES_PER_S}\n')
            f.write(f'min_interval={LOG_SYNC_MIN_INTERVAL}\n')
            f.write(f'max_interval={LOG_SYNC_MAX_INTERVAL}\n')
            f.write('trap_installed=0\n')
        os.replace(tmp, path)

    async def finish(self) -> None:
        """Mark the job done, nudge the syncer past its sleep, wait for the final upload."""
        _active_log_syncers.discard(self)
        self._write_instruction_file(self._instruction_file, self._log_path, self._remote_url, 'done')
        try:
            self._log_copier_proc.send_signal(signal.SIGUSR1)
        except ProcessLookupError:
            pass
        try:
            async with async_timeout.timeout(LOG_SYNC_FINISH_TIMEOUT):
                await self._log_copier_proc.wait()
        except asyncio.TimeoutError:
            log.warning(
                f'log syncer pid={self._log_copier_proc.pid} did not finish in {LOG_SYNC_FINISH_TIMEOUT}s, killing'
            )
            self._log_copier_proc.kill()
            await self._log_copier_proc.wait()
        finally:
            with suppress(FileNotFoundError):
                os.unlink(self._instruction_file)

    async def cancel(self) -> None:
        """Kill the syncer without a final upload (container never ran)."""
        _active_log_syncers.discard(self)
        try:
            self._log_copier_proc.kill()
        except ProcessLookupError:
            pass
        await self._log_copier_proc.wait()
        with suppress(FileNotFoundError):
            os.unlink(self._instruction_file)

    def wakeup(self) -> None:
        """Send SIGUSR1 to interrupt any current sleep without marking the job done."""
        try:
            self._log_copier_proc.send_signal(signal.SIGUSR1)
        except ProcessLookupError:
            pass


def wakeup_all_active_log_syncers() -> None:
    for syncer in _active_log_syncers:
        syncer.wakeup()
