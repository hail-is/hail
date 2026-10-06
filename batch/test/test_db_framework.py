import json

import pytest

from batch.batch_format_version import BatchFormatVersion

from .db_query_checks import assert_hint_kept, assert_scoped, count_row_reads, explain
from .db_seed import LATEST_FORMAT_VERSION, Attempt, Job, JobGroup, Update, preempted_job, seed_batch


async def _fetchall(db, sql, args=None):
    return [r async for r in db.execute_and_fetchall(sql, args)]


async def test_seed_updates_gaps_and_counts(db):
    seeded = await seed_batch(
        db,
        [
            Update(jobs=[Job(), Job(state='Failed'), Job(state='Running')], job_groups=[JobGroup()]),
            # abandoned: reserves 5 ids, uploads 2 jobs and a group, never commits
            Update(
                jobs=[Job(state='Pending', job_group_id=1), Job(state='Pending')],
                job_groups=[JobGroup()],
                committed=False,
                n_reserved_jobs=5,
            ),
            Update(jobs=[Job(job_group_id=1), Job(state='Cancelled', job_group_id=1)]),
            # pending, no jobs uploaded yet
            Update(committed=False, n_reserved_jobs=3),
        ],
    )
    batch_id = seeded.batch_id
    assert [(u.start_job_id, u.n_reserved_jobs, u.job_ids) for u in seeded.updates] == [
        (1, 3, [1, 2, 3]),
        (4, 5, [4, 5]),
        (9, 2, [9, 10]),
        (11, 3, []),
    ]
    assert seeded.committed_job_ids == [1, 2, 3, 9, 10]

    batch = await db.execute_and_fetchone('SELECT n_jobs, state FROM batches WHERE id = %s', (batch_id,))
    assert batch == {'n_jobs': 5, 'state': 'running'}

    job_ids = [r['job_id'] for r in await _fetchall(db, 'SELECT job_id FROM jobs WHERE batch_id = %s', (batch_id,))]
    assert job_ids == [1, 2, 3, 4, 5, 9, 10]

    groups = {
        r['job_group_id']: (r['n_jobs'], r['state'], r['update_id'])
        for r in await _fetchall(
            db, 'SELECT job_group_id, n_jobs, state, update_id FROM job_groups WHERE batch_id = %s', (batch_id,)
        )
    }
    # committed jobs only; group 1 counts 9 and 10 but not the abandoned update's 4
    assert groups == {0: (5, 'running', None), 1: (2, 'complete', 1), 2: (0, 'complete', 2)}

    complete = await db.execute_and_fetchone(
        'SELECT n_completed, n_succeeded, n_failed, n_cancelled FROM job_groups_n_jobs_in_complete_states '
        'WHERE id = %s AND job_group_id = 0',
        (batch_id,),
    )
    assert complete == {'n_completed': 4, 'n_succeeded': 2, 'n_failed': 1, 'n_cancelled': 1}

    # staging rows exist for every update with jobs (including the abandoned one), recursively per ancestor
    # (n_jobs, n_ready_jobs): only update 1's parentless jobs start Ready, as in the front end
    staging = {
        (r['update_id'], r['job_group_id']): (r['n_jobs'], r['n_ready_jobs'])
        for r in await _fetchall(
            db,
            'SELECT update_id, job_group_id, n_jobs, n_ready_jobs FROM job_groups_inst_coll_staging '
            'WHERE batch_id = %s',
            (batch_id,),
        )
    }
    assert staging == {(1, 0): (3, 3), (2, 0): (2, 0), (2, 1): (1, 0), (3, 0): (2, 0), (3, 1): (2, 0)}


async def test_seed_rejects_scheduled_jobs_in_uncommitted_updates(db):
    with pytest.raises(AssertionError, match="state='Pending'"):
        await seed_batch(db, [Update(jobs=[Job()]), Update(jobs=[Job()], committed=False)])
    with pytest.raises(AssertionError, match="state='Ready'"):
        await seed_batch(db, [Update(jobs=[preempted_job()], committed=False)])


async def test_seed_nested_groups(db):
    # 0 -> 1 -> 2 -> 3, plus 12 empty siblings under 1
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[Job(job_group_id=3), Job(job_group_id=2), Job(job_group_id=4)],
                job_groups=[JobGroup(), JobGroup(parent_id=1), JobGroup(parent_id=2)]
                + [JobGroup(parent_id=1) for _ in range(12)],
            )
        ],
    )
    batch_id = seeded.batch_id
    ancestors = await _fetchall(
        db,
        'SELECT ancestor_id, level FROM job_group_self_and_ancestors WHERE batch_id = %s AND job_group_id = 3 '
        'ORDER BY level',
        (batch_id,),
    )
    assert [(r['ancestor_id'], r['level']) for r in ancestors] == [(3, 0), (2, 1), (1, 2), (0, 3)]
    n_jobs = {
        r['job_group_id']: r['n_jobs']
        for r in await _fetchall(db, 'SELECT job_group_id, n_jobs FROM job_groups WHERE batch_id = %s', (batch_id,))
    }
    assert (n_jobs[0], n_jobs[1], n_jobs[2], n_jobs[3], n_jobs[4], n_jobs[5]) == (3, 3, 2, 1, 1, 0)
    assert len(n_jobs) == 16


async def test_seed_preempted_jobs_and_exit_codes(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    preempted_job(),
                    preempted_job(state='Cancelled'),
                    Job(state='Error'),
                    Job(state='Failed', exit_code=137),
                    Job(name=None),
                ]
            )
        ],
    )
    batch_id = seeded.batch_id
    jobs = {
        r['job_id']: r
        for r in await _fetchall(
            db, 'SELECT job_id, state, attempt_id, status FROM jobs WHERE batch_id = %s', (batch_id,)
        )
    }
    assert (jobs[1]['state'], jobs[1]['attempt_id']) == ('Ready', None)
    assert (jobs[2]['state'], jobs[2]['attempt_id']) == ('Cancelled', None)
    assert json.loads(jobs[3]['status'])[0] is None
    assert json.loads(jobs[4]['status'])[0] == 137

    attempts = await _fetchall(
        db, 'SELECT job_id, reason, end_time FROM attempts WHERE batch_id = %s AND job_id IN (1, 2)', (batch_id,)
    )
    assert {(r['job_id'], r['reason']) for r in attempts} == {(1, 'preempted'), (2, 'preempted')}

    names = await _fetchall(db, "SELECT job_id FROM job_attributes WHERE batch_id = %s AND `key` = 'name'", (batch_id,))
    assert [r['job_id'] for r in names] == [1, 2, 3, 4]


@pytest.mark.parametrize('format_version', [1, LATEST_FORMAT_VERSION])
async def test_seed_status_reads_back_through_production_decoder(db, format_version):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[Job(state='Success'), Job(state='Failed', exit_code=2), Job(state='Error'), Job(state='Running')]
            )
        ],
        format_version=format_version,
    )
    rows = await _fetchall(
        db,
        'SELECT batches.format_version, jobs.status FROM jobs JOIN batches ON batches.id = jobs.batch_id '
        'WHERE jobs.batch_id = %s ORDER BY jobs.job_id',
        (seeded.batch_id,),
    )
    assert {r['format_version'] for r in rows} == {format_version}
    decoder = BatchFormatVersion(format_version)
    decoded = [
        tuple(decoder.get_status_exit_code_duration(json.loads(r['status']))) if r['status'] is not None else None
        for r in rows
    ]
    # generated attempts finish 5s after they start; an Error job has no exit code but still a duration
    assert decoded == [(0, 5_000), (2, 5_000), (None, 5_000), None]


async def test_seed_resources_aggregated_by_trigger(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    Job(
                        job_group_id=1,
                        attempts=[
                            Attempt('a', start_time=0, end_time=100, resources={'seed/compute/1': 2}),
                            Attempt('b', start_time=200, end_time=500, resources={'seed/compute/1': 2}),
                        ],
                    ),
                    Job(state='Ready'),
                ],
                job_groups=[JobGroup()],
            )
        ],
    )
    batch_id = seeded.batch_id
    job_usage = await _fetchall(
        db, 'SELECT job_id, `usage` FROM aggregated_job_resources_v3 WHERE batch_id = %s', (batch_id,)
    )
    assert [(r['job_id'], r['usage']) for r in job_usage] == [(1, 2 * 100 + 2 * 300)]
    group_usage = await _fetchall(
        db,
        'SELECT job_group_id, CAST(SUM(`usage`) AS SIGNED) AS u FROM aggregated_job_group_resources_v3 '
        'WHERE batch_id = %s GROUP BY job_group_id',
        (batch_id,),
    )
    assert {r['job_group_id']: r['u'] for r in group_usage} == {0: 800, 1: 800}


async def test_row_reads_bound_scoped_and_unscoped_queries(db, noise_batch):
    seeded = await seed_batch(db, [Update(jobs=[Job() for _ in range(10)])])

    rows, reads = await count_row_reads(
        db, 'SELECT job_id FROM jobs WHERE batch_id = %s AND job_id = %s', (seeded.batch_id, 3)
    )
    assert [r['job_id'] for r in rows] == [3]
    assert reads <= 2

    rows, reads = await count_row_reads(
        db, 'SELECT job_id FROM jobs WHERE batch_id = %s ORDER BY job_id LIMIT 4', (seeded.batch_id,)
    )
    assert len(rows) == 4
    assert reads <= 5

    # an unindexed predicate reads every job in the database, the noise batch's included
    rows, reads = await count_row_reads(db, 'SELECT job_id FROM jobs WHERE spec LIKE %s', ('%no-such-spec%',))
    assert rows == []
    assert reads >= len(noise_batch.committed_job_ids)


async def test_explain_scoped_query_passes(db, noise_batch):
    plan = await explain(
        db,
        """
SELECT /*+ MAX_EXECUTION_TIME(1000) */ jobs.job_id,
  (SELECT `value` FROM job_attributes
   WHERE job_attributes.batch_id = jobs.batch_id AND job_attributes.job_id = jobs.job_id AND `key` = 'name') AS name
FROM jobs FORCE INDEX (PRIMARY)
WHERE jobs.batch_id = %s AND jobs.job_id >= %s
ORDER BY jobs.job_id
LIMIT 50;
""",
        (noise_batch.batch_id, 100),
    )
    # Without FORCE INDEX, MySQL picks jobs_batch_id_update_id here and filesorts, which assert_scoped rejects.
    assert {a.table for a in plan.accesses} == {'jobs', 'job_attributes'}
    assert_scoped(plan)
    assert_hint_kept(plan)


@pytest.mark.usefixtures('noise_batch')  # its rows make the plan realistic
async def test_explain_detects_full_scan(db):
    plan = await explain(db, "SELECT job_id FROM jobs WHERE state = 'Running' AND cores_mcpu + 0 = %s", (1000,))
    with pytest.raises(AssertionError, match='full scan'):
        assert_scoped(plan)


@pytest.mark.usefixtures('noise_batch')  # its rows make the plan realistic
async def test_explain_finds_per_batch_tables_from_schema(db):
    plan = await explain(db, 'SELECT 1')
    columns = plan.batch_scope_columns
    # tables whose batch column is named id, found through their foreign keys to batches.id
    assert columns['batches'] == 'id'
    assert columns['job_groups_n_jobs_in_complete_states'] == 'id'
    assert columns['job_groups_cancelled'] == 'id'
    for table in ('jobs', 'jobs_telemetry', 'job_group_attributes', 'attempt_resources', 'batch_updates'):
        assert columns[table] == 'batch_id', table
    # tables that aren't per batch
    for table in ('billing_projects', 'resources', 'instances', 'user_inst_coll_resources'):
        assert table not in columns, table

    plan = await explain(db, "SELECT id FROM batches WHERE billing_project + '' = %s", ('test',))
    with pytest.raises(AssertionError, match='batches: full scan'):
        assert_scoped(plan)

    plan = await explain(
        db,
        'SELECT n_completed FROM job_groups_n_jobs_in_complete_states WHERE id = %s AND job_group_id = 0',
        (1,),
    )
    assert_scoped(plan)


@pytest.mark.usefixtures('noise_batch')  # its rows make the plan realistic
async def test_explain_resolves_aliases(db):
    sql = """
SELECT j.job_id, latest_attempt.start_time
FROM jobs AS j FORCE INDEX (PRIMARY)
LEFT JOIN attempts AS latest_attempt
  ON latest_attempt.batch_id = j.batch_id AND latest_attempt.job_id = j.job_id AND latest_attempt.attempt_id = j.attempt_id
WHERE j.batch_id = %s
ORDER BY j.job_id
LIMIT 10;
"""
    plan = await explain(db, sql, (1,))
    with pytest.raises(AssertionError, match='latest_attempt: not a table'):
        assert_scoped(plan)
    assert_scoped(plan, aliases={'j': 'jobs', 'latest_attempt': 'attempts'})

    # an aliased scoped table is checked like the table itself
    plan = await explain(db, "SELECT a.job_id FROM attempts AS a WHERE a.reason + '' = %s", ('preempted',))
    with pytest.raises(AssertionError, match=r'a \(alias of attempts\): full scan'):
        assert_scoped(plan, aliases={'a': 'attempts'})


@pytest.mark.usefixtures('noise_batch')  # its rows make the plan realistic
async def test_explain_detects_index_without_batch_id(db):
    plan = await explain(
        db, "SELECT batch_id, job_id FROM job_attributes WHERE `key` = 'shard' AND `value` = %s", ('3',)
    )
    with pytest.raises(AssertionError, match='lacks batch_id|full scan'):
        assert_scoped(plan)


async def test_explain_detects_filesort(db, noise_batch):
    plan = await explain(
        db, 'SELECT job_id FROM jobs WHERE batch_id = %s ORDER BY cores_mcpu LIMIT 10', (noise_batch.batch_id,)
    )
    with pytest.raises(AssertionError, match='filesort'):
        assert_scoped(plan)
    assert_scoped(plan, allow_filesort=True)


async def test_explain_detects_uncorrelated_subquery(db, noise_batch):
    plan = await explain(
        db,
        """
SELECT job_id FROM jobs
WHERE batch_id = %s AND job_id IN (SELECT job_id FROM job_parents WHERE parent_id + 0 = 1)
LIMIT 10;
""",
        (noise_batch.batch_id,),
    )
    with pytest.raises(AssertionError, match='materialized|non-dependent|lacks batch_id|full scan'):
        assert_scoped(plan)


async def test_hint_check_detects_misplaced_hint(db, noise_batch):
    plan = await explain(
        db,
        'SELECT job_id FROM jobs WHERE batch_id = %s AND job_id IN '
        '(SELECT /*+ MAX_EXECUTION_TIME(1000) */ job_id FROM job_parents WHERE batch_id = %s)',
        (noise_batch.batch_id, noise_batch.batch_id),
    )
    with pytest.raises(AssertionError):
        assert_hint_kept(plan)

    plan = await explain(db, 'SELECT job_id FROM jobs WHERE batch_id = %s', (noise_batch.batch_id,))
    with pytest.raises(AssertionError):
        assert_hint_kept(plan)


async def test_seed_n_attempts(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    Job(state='Success', n_attempts=3),
                    Job(state='Running', n_attempts=2),
                    preempted_job(n_attempts=2),
                ]
            )
        ],
    )
    attempts = await _fetchall(
        db,
        'SELECT job_id, attempt_id, reason, end_time FROM attempts WHERE batch_id = %s ORDER BY job_id, start_time',
        (seeded.batch_id,),
    )
    assert [(r['job_id'], r['attempt_id'], r['reason']) for r in attempts] == [
        (1, 'att-1', 'preempted'),
        (1, 'att-2', 'preempted'),
        (1, 'att-3', 'completed'),
        (2, 'att-1', 'preempted'),
        (2, 'att-2', None),
        (3, 'att-1', 'preempted'),
        (3, 'att-2', 'preempted'),
    ]
    current = await _fetchall(db, 'SELECT job_id, attempt_id FROM jobs WHERE batch_id = %s', (seeded.batch_id,))
    assert [(r['job_id'], r['attempt_id']) for r in current] == [(1, 'att-3'), (2, 'att-2'), (3, None)]


async def test_seed_without_costs(db):
    seeded = await seed_batch(db, [Update(jobs=[Job(), Job(state='Running')])], with_costs=False)
    n_attempts = await db.execute_and_fetchone(
        'SELECT COUNT(*) AS n FROM attempts WHERE batch_id = %s', (seeded.batch_id,)
    )
    assert n_attempts['n'] == 2
    for table in ('attempt_resources', 'aggregated_job_resources_v3', 'aggregated_job_group_resources_v3'):
        row = await db.execute_and_fetchone(
            f'SELECT COUNT(*) AS n FROM {table} WHERE batch_id = %s', (seeded.batch_id,)
        )
        assert row['n'] == 0, table


async def test_seed_writes_jobs_in_chunks(db):
    # more than one JOB_CHUNK_SIZE, and a second update starting mid-chunk
    seeded = await seed_batch(
        db, [Update(jobs=[Job() for _ in range(1500)]), Update(jobs=[Job(state='Failed')] * 10)], with_costs=False
    )
    row = await db.execute_and_fetchone(
        'SELECT COUNT(*) AS n, MIN(job_id) AS lo, MAX(job_id) AS hi FROM jobs WHERE batch_id = %s', (seeded.batch_id,)
    )
    assert (row['n'], row['lo'], row['hi']) == (1510, 1, 1510)
    batch = await db.execute_and_fetchone('SELECT n_jobs, state FROM batches WHERE id = %s', (seeded.batch_id,))
    assert batch == {'n_jobs': 1510, 'state': 'complete'}


async def test_seed_creating_job_holds_unbilled_attempt(db):
    seeded = await seed_batch(db, [Update(jobs=[Job(state='Creating')])])
    job = await db.execute_and_fetchone(
        'SELECT attempt_id FROM jobs WHERE batch_id = %s AND job_id = 1', (seeded.batch_id,)
    )
    assert job['attempt_id'] == 'att-1'
    attempt = await db.execute_and_fetchone(
        'SELECT start_time, rollup_time, end_time FROM attempts WHERE batch_id = %s AND job_id = 1', (seeded.batch_id,)
    )
    assert attempt['start_time'] is not None
    assert attempt['rollup_time'] == attempt['start_time']
    assert attempt['end_time'] is None
    cost = await db.execute_and_fetchone(
        'SELECT COUNT(*) AS n FROM aggregated_job_resources_v3 WHERE batch_id = %s', (seeded.batch_id,)
    )
    assert cost['n'] == 0


async def test_seed_n_pending_parents_counts_unfinished_parents(db):
    seeded = await seed_batch(
        db,
        [
            Update(jobs=[Job(state='Success'), Job(state='Running'), Job(state='Pending', parent_ids=[1, 2])]),
            # uncommitted: the front end writes every parent as pending
            Update(jobs=[Job(state='Pending', parent_ids=[1])], committed=False),
        ],
    )
    rows = await _fetchall(
        db, 'SELECT job_id, n_pending_parents FROM jobs WHERE batch_id = %s ORDER BY job_id', (seeded.batch_id,)
    )
    assert [(r['job_id'], r['n_pending_parents']) for r in rows] == [(1, 0), (2, 0), (3, 1), (4, 1)]
