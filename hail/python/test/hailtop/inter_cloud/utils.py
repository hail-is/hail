import secrets
from typing import Dict

from hailtop.aiotools.fs import AsyncFS


async def fresh_dir(fs: AsyncFS, bases: Dict[str, str], scheme: str):
    token = secrets.token_hex(16)
    dir = f'{bases[scheme]}{token}/'
    await fs.mkdir(dir)
    return dir
