"""JSON payloads for the CI developer and operator status endpoints, and CI's own health checks.

Everything here reads in-memory state and resets when CI restarts.
"""

from typing import Any, Dict, List, Optional

from gear import Database
from hailtop.batch_client.aioclient import Batch

from .github import MAX_CONCURRENT_PR_BATCHES, MergeFailureBatch, WatchedBranch

# The update loop runs every 300s, so a branch that hasn't finished one in this long is stuck
UPDATE_STALE_MS = 15 * 60 * 1000
GITHUB_API_ERROR_RECENT_MS = 10 * 60 * 1000
# Webhooks only arrive when something happens on GitHub, so allow for quiet periods
WEBHOOK_GAP_MS = 24 * 60 * 60 * 1000
# Background loops run every 10s
LOOP_STALE_MS = 5 * 60 * 1000
RATE_LIMIT_LOW_FRACTION = 0.1


def _deploy_batch_id(wb: WatchedBranch) -> Optional[int]:
    return wb.deploy_batch.id if isinstance(wb.deploy_batch, Batch) else None


async def developer_branch_status(wb: WatchedBranch, db: Database, frozen: bool) -> Dict[str, Any]:
    prs = wb.prs or {}
    running = [pr.number for pr in wb.running_prs()]
    queued = []
    merge_queue = []
    for pr in wb.prs_in_merge_priority_order():
        # Builds start during each update, so whatever still needs one is waiting for a slot
        if await pr.needs_build(db, on_deck=pr is wb.merge_candidate):
            queued.append({'pr': pr.number, 'reason': pr.pending_build_reason})
        if await pr.could_be_merge_candidate(db):
            failed = wb.failed_merge(pr)
            merge_queue.append({
                'pr': pr.number,
                'blockers': pr.merge_blockers(),
                'failures': failed.failures if failed else 0,
                'backoff_until_ms': failed.retry_after_ms if failed else None,
            })

    last_update = None
    if wb.last_update is not None:
        last_update = {
            'started_ms': wb.last_update.started_ms,
            'finished_ms': wb.last_update.finished_ms,
            'duration_s': (wb.last_update.finished_ms - wb.last_update.started_ms) / 1000,
        }

    merge_candidate = wb.merge_candidate.number if wb.merge_candidate else None
    return {
        'branch': wb.branch.name,
        'sha': wb.sha,
        'deploy_state': wb.deploy_state,
        'deploy_batch_id': _deploy_batch_id(wb),
        'merge_candidate': merge_candidate,
        'n_prs': len(prs),
        'last_update': last_update,
        'merge_paused_reason': wb.merge_paused_reason(frozen),
        'pending_merge': (
            {'pr': wb.pending_merge.pr_number, 'requested_at_ms': wb.pending_merge.requested_at_ms}
            if wb.pending_merge
            else None
        ),
        'build_slots': {
            # The merge candidate bypasses the cap, so used > max is normal
            'used': len(running),
            'max': MAX_CONCURRENT_PR_BATCHES,
            'running': running,
            'on_deck': merge_candidate,
            'queued': queued,
        },
        'merge_queue': merge_queue,
    }


def _rate_limits(wb: WatchedBranch) -> Dict[str, Any]:
    return {
        'rest': wb.rest_rate_limit._asdict() if wb.rest_rate_limit else None,
        'graphql': wb.graphql_rate_limit._asdict() if wb.graphql_rate_limit else None,
    }


def operator_branch_status(wb: WatchedBranch) -> Dict[str, Any]:
    deploy_exception = None
    if isinstance(wb.deploy_batch, MergeFailureBatch):
        exc = wb.deploy_batch.exception
        deploy_exception = f'{type(exc).__name__}: {exc}'
    return {
        'branch': wb.branch.name,
        'pip_version': wb.pip_version,
        'updating': wb.updating,
        'update_started_ms': wb.last_update_started_ms if wb.updating else None,
        'dirty': [
            name
            for name, changed in [
                ('github', wb.github_changed),
                ('batch', wb.batch_changed),
                ('state', wb.state_changed),
            ]
            if changed
        ],
        'last_update_error': wb.last_update.error if wb.last_update else None,
        'rate_limit': _rate_limits(wb),
        'deploy_exception': deploy_exception,
    }


def health_problems(
    watched_branches: List[WatchedBranch],
    *,
    now_ms: int,
    started_at_ms: int,
    last_webhook_ms: Dict[str, int],
    loop_last_success_ms: Dict[str, int],
    expected_loops: List[str],
) -> List[str]:
    """Reasons CI itself looks unhealthy; empty if it looks fine."""
    problems = []

    for wb in watched_branches:
        name = wb.branch.name
        last_finished_ms = wb.last_update.finished_ms if wb.last_update else started_at_ms
        if now_ms - last_finished_ms > UPDATE_STALE_MS:
            problems.append(f'{name}: no update finished in {(now_ms - last_finished_ms) // 60_000} minutes')
        if wb.last_update is not None and wb.last_update.error is not None:
            problems.append(f'{name}: last update failed')
        err = wb.last_github_api_error
        if err is not None and now_ms - err.at_ms < GITHUB_API_ERROR_RECENT_MS:
            problems.append(f'{name}: GitHub API error during {err.operation}')
        for api, rl in [('REST', wb.rest_rate_limit), ('GraphQL', wb.graphql_rate_limit)]:
            if rl is not None and rl.remaining < rl.limit * RATE_LIMIT_LOW_FRACTION:
                problems.append(f'{name}: GitHub {api} rate limit low ({rl.remaining}/{rl.limit})')

    last_webhook = max(last_webhook_ms.values(), default=started_at_ms)
    if now_ms - last_webhook > WEBHOOK_GAP_MS:
        problems.append(f'no GitHub webhook received in {(now_ms - last_webhook) // 3_600_000} hours')

    for loop in expected_loops:
        last = loop_last_success_ms.get(loop, started_at_ms)
        if now_ms - last > LOOP_STALE_MS:
            problems.append(f'background loop {loop} has not succeeded in {(now_ms - last) // 60_000} minutes')

    return problems


def latest_github_api_error(watched_branches: List[WatchedBranch]) -> Optional[Dict[str, Any]]:
    errors = [(wb, wb.last_github_api_error) for wb in watched_branches if wb.last_github_api_error]
    if not errors:
        return None
    wb, err = max(errors, key=lambda e: e[1].at_ms)
    return {'branch': wb.branch.name, **err._asdict()}
