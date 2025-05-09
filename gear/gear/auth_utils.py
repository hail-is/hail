import os
import secrets
from typing import Optional

from hailtop.auth import session_id_encode_to_str

from .database import Database

# Default value: 1800 seconds (30 minutes)
try:
    MAX_AGE_SECS: int = int(os.environ.get("SESSION_MAX_AGE_SECS", "1800"))
except Exception as exc:
    raise ValueError("Unable to interpret SESSION_MAX_AGE_SECS as an integer.") from exc


async def insert_user(db, spec):
    assert all(k in spec for k in ('state', 'username'))

    return await db.execute_insertone(
        f"""
INSERT INTO users ({', '.join(spec.keys())})
VALUES ({', '.join([f'%({k})s' for k in spec.keys()])})
""",
        spec,
    )


async def create_session(db: Database, user_id: int, max_age_secs: Optional[int] = MAX_AGE_SECS) -> str:
    session_id = session_id_encode_to_str(secrets.token_bytes(32))
    await db.just_execute(
        'INSERT INTO sessions (session_id, user_id, max_age_secs) VALUES (%s, %s, %s);',
        (session_id, user_id, max_age_secs),
    )
    return session_id
