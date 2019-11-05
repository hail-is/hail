import asyncio


class CalledProcessError(Exception):
    def __init__(self, command, returncode, outerr):
        super().__init__()
        self.command = command
        self.returncode = returncode
        self.outerr = outerr

    def __str__(self):
        return (f'Command {self.command} returned non-zero exit status {self.returncode}.'
                f' Output:\n{self.outerr}')


async def check_shell_output(script):
    proc = await asyncio.create_subprocess_exec(
        '/bin/bash', '-c', script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    outerr = await proc.communicate()
    if proc.returncode != 0:
        raise CalledProcessError(script, proc.returncode, outerr)
    return outerr


async def check_shell(script):
    # discard output
    await check_shell_output(script)
