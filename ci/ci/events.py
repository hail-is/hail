import logging
from enum import Enum
from typing import Optional

from gear import Database

log = logging.getLogger('ci')

EVENT_RETENTION_DAYS = 90
MAX_MESSAGE_LENGTH = 10_000
_CLEANUP_BATCH_SIZE = 1000


# Mirrors the ENUM on ci_events.event; adding a value needs a migration.
class CIEvent(Enum):
    MERGE_REQUESTED = 'merge_requested'
    MERGE_PENDING = 'merge_pending'
    MERGE_SUCCEEDED = 'merge_succeeded'
    MERGE_FAILED = 'merge_failed'
    MERGE_EXPIRED = 'merge_expired'
    MERGE_CLEARED = 'merge_cleared'
    MERGE_RESTORED = 'merge_restored'
    BUILD_REQUESTED = 'build_requested'
    BUILD_STARTED = 'build_started'
    BUILD_START_FAILED = 'build_start_failed'
    BUILD_CANCELLED = 'build_cancelled'
    RETRY_REQUESTED = 'retry_requested'
    SHA_AUTHORIZED = 'sha_authorized'
    FROZEN = 'frozen'
    UNFROZEN = 'unfrozen'
    DEV_DEPLOY_REQUESTED = 'dev_deploy_requested'
    UPDATE_TRIGGERED = 'update_triggered'
    TARGET_MOVED = 'target_moved'
    DEPLOY_STARTED = 'deploy_started'
    DEPLOY_START_FAILED = 'deploy_start_failed'
    DEPLOY_FINISHED = 'deploy_finished'
    DEPLOY_FAILURE_ALERTED = 'deploy_failure_alerted'
    DEPLOY_STATUS_FAILED = 'deploy_status_failed'
    REVIEWER_ASSIGNED = 'reviewer_assigned'
    REVIEWER_ASSIGN_FAILED = 'reviewer_assign_failed'
    STATUS_POST_FAILED = 'status_post_failed'
    CI_STARTED = 'ci_started'
    UPDATE_FAILED = 'update_failed'
    NAMESPACE_CREATED = 'namespace_created'
    NAMESPACE_DELETED = 'namespace_deleted'
    SERVICE_ADDED = 'service_added'
    SERVICE_EDITED = 'service_edited'
    SERVICE_DELETED = 'service_deleted'
    NAMESPACE_EXPIRED_CLEANUP = 'namespace_expired_cleanup'


def truncate_message(message: str) -> str:
    # keep the tail: for error output and tracebacks the end is the useful part
    if len(message) <= MAX_MESSAGE_LENGTH:
        return message
    marker = '[truncated]\n'
    return marker + message[-(MAX_MESSAGE_LENGTH - len(marker)) :]


async def insert_event(
    db: Database,
    event: CIEvent,
    *,
    target_branch: Optional[str] = None,
    pr_number: Optional[int] = None,
    username: Optional[str] = None,
    batch_id: Optional[int] = None,
    source_sha: Optional[str] = None,
    target_sha: Optional[str] = None,
    previous_sha: Optional[str] = None,
    namespace: Optional[str] = None,
    service: Optional[str] = None,
    merge_uuid: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    await db.execute_insertone(
        """
INSERT INTO ci_events
  (event, target_branch, pr_number, username, batch_id, source_sha, target_sha, previous_sha,
   namespace, service, merge_uuid, message)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
        (
            event.value,
            target_branch,
            pr_number,
            username,
            batch_id,
            source_sha,
            target_sha,
            previous_sha,
            namespace,
            service,
            merge_uuid,
            truncate_message(message) if message is not None else None,
        ),
    )


async def record_event(db: Database, event: CIEvent, **fields) -> None:
    """Best-effort insert_event: a failure to record is logged and never interrupts CI."""
    try:
        await insert_event(db, event, **fields)
    except Exception:  # pylint: disable=broad-except
        log.exception(f'failed to record CI event {event.value}: {fields}')


async def cleanup_old_events(db: Database) -> None:
    while True:
        n_deleted = await db.execute_update(
            f"""
DELETE FROM ci_events
WHERE time < UTC_TIMESTAMP(3) - INTERVAL {EVENT_RETENTION_DAYS} DAY
LIMIT {_CLEANUP_BATCH_SIZE}
"""
        )
        if n_deleted < _CLEANUP_BATCH_SIZE:
            return
