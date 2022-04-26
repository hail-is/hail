from typing import Tuple, List
import asyncio
import subprocess

from .utils import async_to_blocking


class CalledProcessError(Exception):
    def __init__(self, argv: List[str], returncode: int, outerr: Tuple[bytes, bytes]):
        super().__init__()
        self.argv = argv
        self.returncode = returncode
        self._outerr = outerr
        self.stdout = outerr[0]
        self.stderr = outerr[1]

    def __str__(self) -> str:
        return (f'Command {self.argv} returned non-zero exit status {self.returncode}.'
                f' Output:\n{self._outerr}')


async def check_exec_output(command: str,
                            *args: str,
                            echo: bool = False
                            ) -> Tuple[bytes, bytes]:
    if echo:
        print([command, *args])
    proc = await asyncio.create_subprocess_exec(
        command, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    outerr = await proc.communicate()
    assert proc.returncode is not None
    if proc.returncode != 0:
        raise CalledProcessError([command, *args], proc.returncode, outerr)
    return outerr


async def check_shell_output(script: str, echo: bool = False) -> Tuple[bytes, bytes]:
    return await check_exec_output('/bin/bash', '-c', script, echo=echo)


async def check_shell(script: str, echo: bool = False) -> None:
    await check_shell_output(script, echo)


def sync_check_shell_output(script: str, echo=False) -> Tuple[bytes, bytes]:
    return async_to_blocking(check_shell_output(script, echo))


def sync_check_shell(script: str, echo=False) -> None:
    sync_check_shell_output(script, echo)


def sync_check_exec(*command_args: str, echo: bool = False, capture_output: bool = False) -> None:
    if echo:
        print(command_args)
    try:
        subprocess.run(command_args, check=True, capture_output=capture_output)
    except subprocess.CalledProcessError as e:
        raise CalledProcessError(list(command_args), e.returncode, (e.stdout, e.stderr)) from e
