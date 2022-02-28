from typing import Tuple, List, Optional
import asyncio

from .utils import async_to_blocking


class CalledProcessError(Exception):
    def __init__(self, argv: List[str], returncode: int, outerr: Optional[Tuple[bytes, bytes]]):
        super().__init__()
        self.argv = argv
        self.returncode = returncode
        self._outerr = outerr
        self.stdout = outerr[0] if outerr else b''
        self.stderr = outerr[1] if outerr else b''

    def __str__(self) -> str:
        s = f'Command {self.argv} returned non-zero exit status {self.returncode}.'
        if self._outerr:
            s += f'\n Output:\n{self._outerr}'
        else:
            s += f'\n No output available'
        return s


async def check_exec_inherit_output_streams(command: str, *args: str, echo: bool = False) -> None:
    if echo:
        print([command, *args])
    proc = await asyncio.create_subprocess_exec(command, *args)
    await proc.wait()
    assert proc.returncode is not None
    if proc.returncode != 0:
        raise CalledProcessError([command, *args], proc.returncode, None)


async def check_exec_output(command: str, *args: str, echo: bool = False) -> Tuple[bytes, bytes]:
    if echo:
        print([command, *args])
    proc = await asyncio.create_subprocess_exec(command, *args,
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    outerr = await proc.communicate()
    assert proc.returncode is not None
    if proc.returncode != 0:
        raise CalledProcessError([command, *args], proc.returncode, outerr)
    return outerr


async def check_shell_output(script: str, echo: bool = False) -> Tuple[bytes, bytes]:
    return await check_exec_output('/bin/bash', '-c', script, echo=echo)


async def check_shell(script: str, echo: bool = False, inherit_std_out_err: bool = False) -> None:
    if inherit_std_out_err:
        await check_exec_inherit_output_streams('/bin/bash', '-c', script, echo=echo)
    else:
        # Use version that collects stdout/stderr for error reporting
        await check_shell_output(script, echo=echo)


def sync_check_exec_output(command: str, *args: str, echo: bool = False) -> Tuple[bytes, bytes]:
    return async_to_blocking(check_exec_output(command, *args, echo=echo))


def sync_check_exec(command: str, *args: str,echo: bool = False):
    sync_check_exec_output(command, *args, echo=echo)


def sync_check_shell_output(script: str, echo=False) -> Tuple[bytes, bytes]:
    return async_to_blocking(check_shell_output(script, echo))


def sync_check_shell(script: str, echo=False, inherit_std_out_err: bool = False) -> None:
    async_to_blocking(check_shell(script, echo, inherit_std_out_err=inherit_std_out_err))
