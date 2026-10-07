"""Seed consistent batch rows directly into a real-migration batch database, for DB-layer tests.

Rows are written the way the front end and driver would leave them, without going through
``commit_batch_update`` or the scheduling procedures, so a test can put jobs in any final state. The
tables the read paths use are kept consistent with each other (``batches``, ``batch_updates``,
``job_groups``, ``job_group_self_and_ancestors``, ``job_groups_n_jobs_in_complete_states``,
``job_groups_inst_coll_staging``, ``jobs``, ``job_attributes``, ``job_parents``, ``attempts``,
``attempt_resources``, and, through the ``attempt_resources`` trigger, the aggregated resource tables).
Scheduler bookkeeping (``user_inst_coll_resources``, ``job_group_inst_coll_cancellable_resources``,
``jobs_telemetry``) is not written, so seeded batches must never be handed to a driver.

Each call to :func:`seed_batch` creates a new batch with its own auto-increment id; the database is shared
across the test session, so tests must only look at their own batch ids.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from batch.batch_format_version import BatchFormatVersion
from batch.globals import BATCH_FORMAT_VERSION as LATEST_FORMAT_VERSION
from gear import Database
from hailtop.batch_client.globals import MAX_JOB_GROUPS_DEPTH

ROOT_JOB_GROUP_ID = 0
BILLING_PROJECT = 'test'
USER = 'test'

COMPLETE_STATES = ('Success', 'Failed', 'Error', 'Cancelled')
ALL_STATES = ('Pending', 'Ready', 'Creating', 'Running', *COMPLETE_STATES)

# Resources the seeder bills against. Real rows exist too, but these keep seeded costs independent of them.
SEED_RESOURCES = {'seed/compute/1': 0.001, 'seed/memory/1': 0.0001, 'seed/disk/1': 0.00001}

_UNSET: Any = object()

T0 = 1_700_000_000_000

# Jobs are written in transactions of at most this many, like the client's job bunches (MAX_BUNCH_SIZE).
JOB_CHUNK_SIZE = 1024

# States whose job holds an attempt (jobs.attempt_id): Creating and Running are in progress, the rest finished.
_ATTEMPTED_STATES = ('Creating', 'Running', 'Success', 'Failed', 'Error')


@dataclass
class Attempt:
    attempt_id: str
    start_time: Optional[int]
    end_time: Optional[int] = None
    # Defaults to end_time, or start_time + 1s for an attempt that's still running.
    rollup_time: Optional[int] = _UNSET
    reason: Optional[str] = None
    # Defaults to seed-{the job's inst_coll}; the seeder writes a matching instances row (attempts has a foreign key).
    instance_name: Optional[str] = _UNSET
    # resource name -> quantity; usage is quantity * (rollup_time - start_time).
    resources: Dict[str, int] = field(default_factory=lambda: {'seed/compute/1': 1000, 'seed/memory/1': 3840})


@dataclass
class Job:
    """One job. ``job_group_id`` is absolute (0 is the root, sub-groups are numbered in creation order)."""

    state: str = 'Success'
    job_group_id: int = ROOT_JOB_GROUP_ID
    name: Optional[str] = _UNSET  # defaults to 'job-{job_id}'; None means no name attribute
    attributes: Dict[str, str] = field(default_factory=dict)
    parent_ids: Sequence[int] = ()
    # Defaults: 0 for Success, 1 for Failed, None (stored as [null, …]) for Error, no status otherwise.
    exit_code: Optional[int] = _UNSET
    # How many attempts to generate: all but the last were preempted. The last is in progress for a Creating or
    # Running job, finished for Success/Failed/Error, and preempted for any other state. Defaults to 1 for
    # Creating, Running, Success, Failed and Error, 0 otherwise.
    n_attempts: Optional[int] = None
    # Explicit attempts instead of generated ones (not with n_attempts).
    attempts: Optional[List[Attempt]] = None
    cancelled: bool = False
    inst_coll: str = 'standard'
    cores_mcpu: int = 1000


@dataclass
class JobGroup:
    parent_id: int = ROOT_JOB_GROUP_ID


@dataclass
class Update:
    """A batch update. ``committed=False`` leaves it pending; one followed by later updates is abandoned.

    In a committed update a job must be Pending exactly when one of its parents hasn't finished, as in production.

    Jobs in an uncommitted update can't have been scheduled: they must be in their initial state (``Ready`` in
    update 1 with no parents, ``Pending`` otherwise) with no attempts, so they never carry cost.

    ``n_reserved_jobs`` (default ``len(jobs)``) is the id range the update claims. Reserving more than
    the jobs written leaves an id gap, as a partly uploaded update does.
    """

    jobs: List[Job] = field(default_factory=list)
    job_groups: List[JobGroup] = field(default_factory=list)
    committed: bool = True
    n_reserved_jobs: Optional[int] = None


@dataclass
class SeededUpdate:
    update_id: int
    start_job_id: int
    n_reserved_jobs: int
    job_ids: List[int]
    start_job_group_id: int
    job_group_ids: List[int]
    committed: bool


@dataclass
class SeededBatch:
    batch_id: int
    format_version: int
    updates: List[SeededUpdate]
    jobs: Dict[int, Job]  # job_id -> spec
    job_group_parents: Dict[int, Optional[int]]  # job_group_id -> parent (None for the root)

    @property
    def committed_job_ids(self) -> List[int]:
        return sorted(j for u in self.updates if u.committed for j in u.job_ids)

    def ancestors(self, job_group_id: int) -> List[int]:
        """Self and ancestors, self first."""
        result = []
        g: Optional[int] = job_group_id
        while g is not None:
            result.append(g)
            g = self.job_group_parents[g]
        return result


def preempted_job(state: str = 'Ready', n_attempts: int = 1, **kwargs) -> Job:
    """A job whose attempts were all preempted, now back in ``state`` with ``jobs.attempt_id = NULL``.

    ``state='Cancelled'`` gives a job that was preempted and then cancelled: it has a start time from
    the preempted attempt but no end time.
    """
    assert state not in _ATTEMPTED_STATES, state
    return Job(state=state, n_attempts=n_attempts, **kwargs)


async def ensure_seed_resources(db: Database) -> Dict[str, int]:
    """Insert SEED_RESOURCES if missing and return their ids."""
    await db.execute_many(
        'INSERT INTO resources (resource, rate) VALUES (%s, %s) ON DUPLICATE KEY UPDATE rate = rate',
        list(SEED_RESOURCES.items()),
    )
    await db.execute_update(
        'UPDATE resources SET deduped_resource_id = resource_id WHERE resource LIKE %s AND deduped_resource_id IS NULL',
        ('seed/%',),
    )
    return {
        r['resource']: r['resource_id']
        async for r in db.execute_and_fetchall(
            'SELECT resource, resource_id FROM resources WHERE resource LIKE %s', ('seed/%',)
        )
    }


def _attempts(job_id: int, job: Job) -> List[Attempt]:
    if job.attempts is not None:
        assert job.n_attempts is None, 'give attempts or n_attempts, not both'
        return job.attempts
    n = job.n_attempts if job.n_attempts is not None else int(job.state in _ATTEMPTED_STATES)
    base = T0 + job_id * 100_000
    attempts = []
    for k in range(n):
        start = base + k * 10_000
        attempt_id = f'att-{k + 1}'
        if k < n - 1 or job.state not in _ATTEMPTED_STATES:
            attempts.append(Attempt(attempt_id, start_time=start, end_time=start + 5_000, reason='preempted'))
        elif job.state == 'Creating':
            # mark_job_creating adds the attempt with rollup_time = start_time: nothing billed yet
            attempts.append(Attempt(attempt_id, start_time=start, rollup_time=start))
        elif job.state == 'Running':
            attempts.append(Attempt(attempt_id, start_time=start))
        else:
            reason = 'error' if job.state == 'Error' else 'completed'
            attempts.append(Attempt(attempt_id, start_time=start, end_time=start + 5_000, reason=reason))
    return attempts


def _rollup_time(a: Attempt) -> Optional[int]:
    if a.rollup_time is not _UNSET:
        return a.rollup_time
    if a.end_time is not None:
        return a.end_time
    return a.start_time + 1_000 if a.start_time is not None else None


def _cancelled(seeded: SeededBatch, job: Job, committed: bool) -> bool:
    """jobs.cancelled: as given, or set by production once any parent finished other than Success (on commit,
    and as each parent completes). Nothing has been derived yet in an uncommitted update."""
    if not committed:
        return job.cancelled
    return job.cancelled or any(
        seeded.jobs[p].state in COMPLETE_STATES and seeded.jobs[p].state != 'Success' for p in job.parent_ids
    )


def _unfinished_parents(seeded: SeededBatch, job: Job) -> int:
    return sum(seeded.jobs[p].state not in COMPLETE_STATES for p in job.parent_ids)


def _check_state_matches_parents(seeded: SeededBatch, job_id: int, job: Job):
    """A committed job is Pending exactly while it has unfinished parents: commit_batch_update and each parent's
    completion move it to Ready at zero, and cancellation only acts on Ready jobs (driver/canceller.py)."""
    unfinished = _unfinished_parents(seeded, job)
    if job.state == 'Pending':
        assert unfinished > 0, f'job {job_id} is Pending with no unfinished parents; production would make it Ready'
    else:
        assert unfinished == 0, (
            f'job {job_id} is {job.state} with {unfinished} unfinished parent(s); production keeps it Pending'
        )


def _n_pending_parents(seeded: SeededBatch, job: Job, committed: bool) -> int:
    """The front end writes every parent as pending; committing (and each parent completing) counts down to the
    parents that haven't finished. Only a Pending job has any left: the rest became Ready at zero."""
    if job.state != 'Pending':
        return 0
    if not committed:
        return len(job.parent_ids)
    return _unfinished_parents(seeded, job)


def _default_exit_code(state: str) -> Optional[int]:
    return {'Success': 0, 'Failed': 1}.get(state)


def _db_status(format_version: int, job: Job, attempts: List[Attempt]) -> Optional[str]:
    """The worker's status for a finished job, encoded by the production encoder (BatchFormatVersion.db_status):
    the full dict for format version 1, ``[exit_code, duration]`` after."""
    if job.state not in ('Success', 'Failed', 'Error'):
        return None
    ec = _default_exit_code(job.state) if job.exit_code is _UNSET else job.exit_code
    last = attempts[-1] if attempts else None
    start_time = last.start_time if last else None
    end_time = last.end_time if last else None
    duration = end_time - start_time if start_time is not None and end_time is not None else None

    main: Dict[str, Any] = {'error': 'seeded error'} if ec is None else {'container_status': {'exit_code': ec}}
    if format_version == 1:
        # Format-version-1 batches ran when workers wrote status version 1, where duration is the sum of each
        # container's timing.runtime.duration. Only main: input/output containers only ran with files.
        if duration is not None:
            main['timing'] = {'runtime': {'duration': duration}}
        status = {'version': 1, 'state': job.state.lower(), 'container_statuses': {'main': main}}
    else:
        status = {
            'version': 2,
            'state': job.state.lower(),
            'start_time': start_time,
            'end_time': end_time,
            'container_statuses': {'main': main},
        }
    return json.dumps(BatchFormatVersion(format_version).db_status(status))


def _db_spec(format_version: int) -> str:
    if format_version == 1:
        return json.dumps({'image': 'ubuntu:24.04', 'command': ['true'], 'resources': {}})
    if format_version < 5:
        return json.dumps([None, None, 0, 0])
    return json.dumps([None, None, 0, 0, None])


# Every VALUES tuple below is all placeholders: PyMySQL only batches executemany into multi-row INSERTs then,
# and otherwise sends one statement per row.
_INSERT_BATCH_UPDATE = """
INSERT INTO batch_updates (batch_id, update_id, token, start_job_group_id, n_job_groups, start_job_id, n_jobs,
  committed, time_created, time_committed)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""
_INSERT_JOB_GROUP = """
INSERT INTO job_groups (batch_id, job_group_id, `user`, attributes, state, n_jobs, time_created, time_completed,
  update_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""
_INSERT_JOB_GROUP_ANCESTOR = """
INSERT INTO job_group_self_and_ancestors (batch_id, job_group_id, ancestor_id, level) VALUES (%s, %s, %s, %s);
"""
_INSERT_JOB_GROUP_COMPLETE_STATES = """
INSERT INTO job_groups_n_jobs_in_complete_states (id, job_group_id, n_completed, n_succeeded, n_failed, n_cancelled)
VALUES (%s, %s, %s, %s, %s, %s);
"""
_INSERT_JOB = """
INSERT INTO jobs (batch_id, job_id, update_id, job_group_id, state, spec, status, always_run, cores_mcpu,
  n_pending_parents, cancelled, attempt_id, inst_coll)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""
_INSERT_JOB_ATTRIBUTE = 'INSERT INTO job_attributes (batch_id, job_id, `key`, `value`) VALUES (%s, %s, %s, %s);'
_INSERT_JOB_PARENT = 'INSERT INTO job_parents (batch_id, job_id, parent_id) VALUES (%s, %s, %s);'
_INSERT_ATTEMPT = """
INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name, start_time, rollup_time, end_time, reason)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
"""
# Instances are shared across batches, so an existing one is left as it is.
_INSERT_INSTANCE = """
INSERT INTO instances (name, state, token, cores_mcpu, time_created, last_updated, version, location, inst_coll,
  machine_type, preemptible)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE name = name;
"""
# The attempt_resources_after_insert trigger fills the aggregated_*_resources_v3 tables, once per row.
_INSERT_ATTEMPT_RESOURCE = """
INSERT INTO attempt_resources (batch_id, job_id, attempt_id, quantity, resource_id, deduped_resource_id)
VALUES (%s, %s, %s, %s, %s, %s);
"""
_INSERT_STAGING = """
INSERT INTO job_groups_inst_coll_staging (batch_id, update_id, job_group_id, inst_coll, token, n_jobs,
  n_ready_jobs, ready_cores_mcpu)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
"""


async def seed_batch(
    db: Database,
    updates: Sequence[Update],
    *,
    format_version: int = LATEST_FORMAT_VERSION,
    with_costs: bool = True,
) -> SeededBatch:
    """Create a batch with the given updates, in order, and return what was written.

    ``with_costs=False`` skips ``attempt_resources`` (so the jobs have no cost), which is most of the seeding
    time: its trigger runs several statements per row. Attempts and their times are still written. Use it for
    large batches that aren't testing cost; the noise batch keeps the cost tables populated for EXPLAIN.

    Staging rows (``job_groups_inst_coll_staging``) are written per chunk with the chunk's jobs, one token per
    chunk, as the front end writes them per create-jobs request. Committed updates' staging rows are kept:
    production has them until the driver's cleanup loop deletes them, and keeping them is the stricter case for
    a query that must only count pending updates.

    Rows are planned in memory first (an invalid plan deletes the batch row and raises), then written: the
    updates and groups in one transaction, then jobs in transactions of JOB_CHUNK_SIZE (like the client's job
    bunches). A database error part way through writing leaves a partial batch; tests don't reuse ids.
    """
    resource_ids = await ensure_seed_resources(db) if with_costs else {}

    batch_id = await db.execute_insertone(
        """
INSERT INTO batches (userdata, user, billing_project, attributes, n_jobs, time_created, token, state,
  format_version, migrated_batch)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
""",
        ('{}', USER, BILLING_PROJECT, '{}', 0, T0, None, 'running', format_version, True),
    )
    assert batch_id is not None
    seeded = SeededBatch(batch_id, format_version, [], {}, {ROOT_JOB_GROUP_ID: None})

    try:
        update_rows, job_group_rows, job_chunks, instances = _plan(
            seeded, updates, format_version=format_version, with_costs=with_costs, resource_ids=resource_ids
        )
    except BaseException:
        # A plan the seeder rejects leaves nothing behind: so far only the batch row exists.
        await db.execute_update('DELETE FROM batches WHERE id = %s', (batch_id,))
        raise

    n_jobs, n_complete = _counts(seeded)

    def group_state(g):
        return 'complete' if sum(n_complete[g].values()) == n_jobs[g] else 'running'

    async with db.start() as tx:
        if instances:
            await tx.execute_many(
                _INSERT_INSTANCE,
                [
                    (name, 'active', 'seed', 16_000, T0, T0, 1, 'us-central1-a', ic, 'n1-standard-16', True)
                    for name, ic in sorted(instances.items())
                ],
            )
        await tx.execute_many(_INSERT_BATCH_UPDATE, update_rows)
        await tx.execute_many(
            _INSERT_JOB_GROUP,
            [
                (
                    batch_id,
                    g,
                    USER,
                    '{}',
                    group_state(g),
                    n_jobs[g],
                    T0,
                    T0 if group_state(g) == 'complete' else None,
                    update_id,
                )
                for g, update_id in job_group_rows
            ],
        )
        await tx.execute_many(
            _INSERT_JOB_GROUP_ANCESTOR,
            [(batch_id, g, a, level) for g, _ in job_group_rows for level, a in enumerate(seeded.ancestors(g))],
        )
        await tx.execute_many(
            _INSERT_JOB_GROUP_COMPLETE_STATES,
            [
                (batch_id, g, sum(c.values()), c['Success'], c['Failed'] + c['Error'], c['Cancelled'])
                for g, c in n_complete.items()
            ],
        )

    for chunk in job_chunks:
        async with db.start() as tx:
            for sql, key in (
                (_INSERT_JOB, 'jobs'),
                (_INSERT_JOB_ATTRIBUTE, 'attributes'),
                (_INSERT_JOB_PARENT, 'parents'),
                (_INSERT_ATTEMPT, 'attempts'),
                (_INSERT_ATTEMPT_RESOURCE, 'resources'),
                (_INSERT_STAGING, 'staging'),
            ):
                if chunk[key]:
                    await tx.execute_many(sql, chunk[key])

    root_state = group_state(ROOT_JOB_GROUP_ID)
    async with db.start() as tx:
        await tx.execute_update(
            'UPDATE batches SET n_jobs = %s, state = %s, time_completed = %s WHERE id = %s',
            (n_jobs[ROOT_JOB_GROUP_ID], root_state, T0 if root_state == 'complete' else None, batch_id),
        )

    return seeded


def _plan(seeded: SeededBatch, updates: Sequence[Update], *, format_version: int, with_costs: bool, resource_ids):
    """Validate the updates and build every row to write, without touching the database."""
    batch_id = seeded.batch_id
    update_rows = []
    job_group_rows: List[tuple] = [(ROOT_JOB_GROUP_ID, None)]  # (job_group_id, update_id)
    job_chunks: List[Dict[str, list]] = []
    instances: Dict[str, str] = {}  # name -> inst_coll

    next_job_id = 1
    next_job_group_id = 1
    for update_id, update in enumerate(updates, start=1):
        n_reserved = len(update.jobs) if update.n_reserved_jobs is None else update.n_reserved_jobs
        assert n_reserved >= len(update.jobs)
        assert not update.committed or n_reserved == len(update.jobs), 'a committed update writes all its jobs'
        update_rows.append((
            batch_id,
            update_id,
            f'seed-{update_id}',
            next_job_group_id,
            len(update.job_groups),
            next_job_id,
            n_reserved,
            update.committed,
            T0,
            T0 if update.committed else None,
        ))

        job_group_ids = []
        for jg in update.job_groups:
            assert jg.parent_id in seeded.job_group_parents, f'job group parent {jg.parent_id} does not exist'
            # The front end's limit: the parent's own ancestry (itself up to the root) must be at most
            # MAX_JOB_GROUPS_DEPTH rows, so groups nest at most MAX_JOB_GROUPS_DEPTH levels below the root.
            assert len(seeded.ancestors(jg.parent_id)) <= MAX_JOB_GROUPS_DEPTH, (
                f'job group {next_job_group_id} would be nested deeper than MAX_JOB_GROUPS_DEPTH '
                f'({MAX_JOB_GROUPS_DEPTH}) below the root'
            )
            seeded.job_group_parents[next_job_group_id] = jg.parent_id
            job_group_rows.append((next_job_group_id, update_id))
            job_group_ids.append(next_job_group_id)
            next_job_group_id += 1

        job_ids = list(range(next_job_id, next_job_id + len(update.jobs)))
        chunk_staging: List[Dict[tuple, List[Optional[int]]]] = []
        for i, (job_id, job) in enumerate(zip(job_ids, update.jobs)):
            if i % JOB_CHUNK_SIZE == 0:
                job_chunks.append({
                    'jobs': [],
                    'attributes': [],
                    'parents': [],
                    'attempts': [],
                    'resources': [],
                    'staging': [],
                })
                chunk_staging.append({})
            chunk = job_chunks[-1]
            staging = chunk_staging[-1]
            assert job.state in ALL_STATES, job.state
            assert job.job_group_id in seeded.job_group_parents, f'job group {job.job_group_id} does not exist'
            # before registering this job, so it can't be its own parent
            for p in job.parent_ids:
                assert p in seeded.jobs, f'job {job_id}: parent {p} must be an earlier seeded job'
            seeded.jobs[job_id] = job

            attempts = _attempts(job_id, job)
            # The front end starts a job Ready only in update 1 and with no parents; anything later starts
            # Pending, since its parents may be in earlier updates (front_end.py, "always start out as pending").
            initially_ready = update_id == 1 and not job.parent_ids
            if not update.committed:
                initial_state = 'Ready' if initially_ready else 'Pending'
                assert job.state == initial_state and not attempts, (
                    f'job {job_id} is in uncommitted update {update_id}, so it cannot have been scheduled: '
                    f'give it state={initial_state!r} and no attempts'
                )
            else:
                _check_state_matches_parents(seeded, job_id, job)
            # jobs.attempt_id: the last attempt while it holds one; the driver clears it on preemption
            current_attempt_id = attempts[-1].attempt_id if attempts and job.state in _ATTEMPTED_STATES else None
            # production always adds the attempt before a job becomes Creating or Running (mark_job_creating,
            # scheduling)
            assert job.state not in ('Creating', 'Running') or current_attempt_id is not None, (
                f'job {job_id} is {job.state}, so it must have a current attempt'
            )
            chunk['jobs'].append((
                batch_id,
                job_id,
                update_id,
                job.job_group_id,
                job.state,
                _db_spec(format_version),
                _db_status(format_version, job, attempts),
                False,  # always_run
                job.cores_mcpu,
                _n_pending_parents(seeded, job, update.committed),
                _cancelled(seeded, job, update.committed),
                current_attempt_id,
                job.inst_coll,
            ))
            name = f'job-{job_id}' if job.name is _UNSET else job.name
            attributes = {**({'name': name} if name is not None else {}), **job.attributes}
            chunk['attributes'].extend((batch_id, job_id, k, v) for k, v in attributes.items())
            chunk['parents'].extend((batch_id, job_id, p) for p in job.parent_ids)
            for a in attempts:
                instance_name = f'seed-{job.inst_coll}' if a.instance_name is _UNSET else a.instance_name
                if instance_name is not None:
                    assert instances.setdefault(instance_name, job.inst_coll) == job.inst_coll, (
                        f'instance {instance_name} is in two instance collections'
                    )
                chunk['attempts'].append((
                    batch_id,
                    job_id,
                    a.attempt_id,
                    instance_name,
                    a.start_time,
                    _rollup_time(a),
                    a.end_time,
                    a.reason,
                ))
                if with_costs:
                    chunk['resources'].extend(
                        (batch_id, job_id, a.attempt_id, q, resource_ids[r], resource_ids[r])
                        for r, q in a.resources.items()
                    )
            for ancestor in seeded.ancestors(job.job_group_id):
                staging.setdefault((ancestor, job.inst_coll), []).append(job.cores_mcpu if initially_ready else None)

        # Written with each chunk's job rows, recursively (one row per ancestor), one token per chunk: the front
        # end writes them per create-jobs request with a random token.
        for token, (chunk, staging) in enumerate(zip(job_chunks[-len(chunk_staging) :], chunk_staging)):
            chunk['staging'].extend(
                (
                    batch_id,
                    update_id,
                    g,
                    ic,
                    token,
                    len(cores),
                    sum(c is not None for c in cores),
                    sum(c or 0 for c in cores),
                )
                for (g, ic), cores in staging.items()
            )
        seeded.updates.append(
            SeededUpdate(
                update_id,
                next_job_id,
                n_reserved,
                job_ids,
                update_rows[-1][3],
                job_group_ids,
                update.committed,
            )
        )
        next_job_id += n_reserved

    return update_rows, job_group_rows, job_chunks, instances


def _counts(seeded: SeededBatch):
    """The counters commit_batch_update and mark_job_complete would have maintained (committed jobs only)."""
    n_jobs: Dict[int, int] = dict.fromkeys(seeded.job_group_parents, 0)
    n_complete: Dict[int, Dict[str, int]] = {g: dict.fromkeys(COMPLETE_STATES, 0) for g in seeded.job_group_parents}
    for job_id in seeded.committed_job_ids:
        job = seeded.jobs[job_id]
        for g in seeded.ancestors(job.job_group_id):
            n_jobs[g] += 1
            if job.state in COMPLETE_STATES:
                n_complete[g][job.state] += 1
    return n_jobs, n_complete


async def analyze_tables(db: Database):
    """Refresh index statistics for every table, so EXPLAIN reflects the seeded row counts."""
    tables = [
        r['name']
        async for r in db.execute_and_fetchall(
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_type = 'BASE TABLE'"
        )
    ]
    await db.just_execute('ANALYZE TABLE ' + ', '.join(f'`{t}`' for t in tables))
