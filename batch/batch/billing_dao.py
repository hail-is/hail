from typing import Optional

from gear import Database
from hailtop.utils import time_msecs

from .exceptions import BatchOperationAlreadyCompletedError


async def create_quote(
    db: Database,
    name: str,
    cost_object: str,
    actor: str,
    authorized_amount: Optional[float] = None,
    pi_name: Optional[str] = None,
    pm_designee: Optional[str] = None,
    comment: Optional[str] = None,
) -> int:
    """Insert a new quote and log the creation event. Returns the new quote id.

    Raises BatchOperationAlreadyCompletedError if the name already exists.
    """
    async with db.start() as tx:
        row = await tx.execute_and_fetchone('SELECT id FROM quotes WHERE name = %s FOR UPDATE;', (name,))
        if row is not None:
            raise BatchOperationAlreadyCompletedError(f'Quote {name} already exists.', 'info')

        now = time_msecs()
        quote_id = await tx.execute_insertone(
            """
INSERT INTO quotes (name, name_cs, cost_object, authorized_amount, pi_name, pm_designee, time_created)
VALUES (%s, %s, %s, %s, %s, %s, %s);
""",
            (name, name, cost_object, authorized_amount, pi_name, pm_designee, now),
        )
        assert quote_id is not None
        await tx.execute_insertone(
            """
INSERT INTO quote_events (quote_id, timestamp, actor, action, target_user, target_project, detail, comment)
VALUES (%s, %s, %s, %s, NULL, NULL, %s, %s);
""",
            (quote_id, now, actor, 'quote_created', cost_object, comment),
        )
        return quote_id
