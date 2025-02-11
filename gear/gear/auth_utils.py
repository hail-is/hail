import os
import secrets
from typing import Optional

from hailtop.auth import session_id_encode_to_str

from .database import Database

# Default value: 86400 seconds (1 day)
if os.environ["SESSION_MAX_AGE_SECS"] is not None:
    MAX_AGE_SECS = os.environ["SESSION_MAX_AGE_SECS"]
else:
    MAX_AGE_SECS = 86400


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
