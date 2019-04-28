import string
import secrets
import asyncio


class CalledProcessError(Exception):
    def __init__(self, command, returncode):
        super().__init__()
        self.command = command
        self.returncode = returncode

    def __str__(self):
        return f'Command {self.command} returned non-zero exit status {self.returncode}.'


async def check_shell(script):
    proc = await asyncio.create_subprocess_shell(script)
    await proc.wait()
    if proc.returncode != 0:
        raise CalledProcessError(script, proc.returncode)


async def check_shell_output(script):
    proc = await asyncio.create_subprocess_shell(
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    outerr = await proc.communicate()
    if proc.returncode != 0:
        raise CalledProcessError(script, proc.returncode)
    return outerr


def generate_token(size=12):
    assert size > 0
    alpha = string.ascii_lowercase
    alnum = string.ascii_lowercase + string.digits
    return secrets.choice(alpha) + ''.join([secrets.choice(alnum) for _ in range(size - 1)])


def flatten(xxs):
    return [x for xs in xxs for x in xs]


# FIXME move to batch
def update_batch_status(status):
    jobs = status['jobs']

    if any(job['state'] == 'Complete' and job['exit_code'] > 0 for job in jobs):
        state = 'failure'
    elif any(job['state'] == 'Cancelled' for job in jobs):
        state = 'cancelled'
    elif all(job['state'] == 'Complete' and job['exit_code'] == 0 for job in jobs):
        state = 'success'
    else:
        state = 'running'

    if state:
        status['state'] = state

    complete = all(job['state'] in ('Cancelled', 'Complete') for job in jobs)
    status['complete'] = complete
