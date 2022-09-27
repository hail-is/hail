DELIMITER $$

DROP PROCEDURE IF EXISTS commit_batch_update $$
CREATE PROCEDURE commit_batch_update(
  IN in_batch_id BIGINT,
  IN in_update_id INT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_update_committed BOOLEAN;
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE cur_update_start_job_id INT;

  START TRANSACTION;

  SELECT committed, n_jobs INTO cur_update_committed, expected_n_jobs
  FROM batch_updates
  WHERE batch_id = in_batch_id AND update_id = in_update_id
  FOR UPDATE;

  IF cur_update_committed THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0) INTO staging_n_jobs
    FROM batches_inst_coll_staging
    WHERE batch_id = in_batch_id AND update_id = in_update_id
    FOR UPDATE;

    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE batch_id = in_batch_id AND update_id = in_update_id;

      UPDATE batches SET
        `state` = 'running',
        time_completed = NULL,
        n_jobs = n_jobs + expected_n_jobs
      WHERE id = in_batch_id;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;

      IF in_update_id != 1 THEN
        SELECT start_job_id INTO cur_update_start_job_id FROM batch_updates WHERE batch_id = in_batch_id AND update_id = in_update_id;

        UPDATE jobs
          LEFT JOIN (
            SELECT `job_parents`.batch_id, `job_parents`.job_id,
              COALESCE(SUM(1), 0) AS n_parents,
              COALESCE(SUM(state IN ('Pending', 'Ready', 'Creating', 'Running')), 0) AS n_pending_parents,
              COALESCE(SUM(state = 'Success'), 0) AS n_succeeded
            FROM `job_parents`
            LEFT JOIN `jobs` ON jobs.batch_id = `job_parents`.batch_id AND jobs.job_id = `job_parents`.parent_id
            WHERE job_parents.batch_id = in_batch_id AND
              `job_parents`.job_id >= cur_update_start_job_id AND
              `job_parents`.job_id < cur_update_start_job_id + staging_n_jobs
            GROUP BY `job_parents`.batch_id, `job_parents`.job_id
            FOR UPDATE
          ) AS t
            ON jobs.batch_id = t.batch_id AND
               jobs.job_id = t.job_id
          SET jobs.state = IF(COALESCE(t.n_pending_parents, 0) = 0, 'Ready', 'Pending'),
              jobs.n_pending_parents = COALESCE(t.n_pending_parents, 0),
              jobs.cancelled = IF(COALESCE(t.n_succeeded, 0) = COALESCE(t.n_parents - t.n_pending_parents, 0), jobs.cancelled, 1)
          WHERE jobs.batch_id = in_batch_id AND jobs.job_id >= cur_update_start_job_id AND
              jobs.job_id < cur_update_start_job_id + staging_n_jobs;
      END IF;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 1 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS cancel_batch $$
CREATE PROCEDURE cancel_batch(
  IN in_batch_id VARCHAR(100)
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;
  DECLARE cur_n_cancelled_ready_jobs INT;
  DECLARE cur_cancelled_ready_cores_mcpu BIGINT;
  DECLARE cur_n_cancelled_running_jobs INT;
  DECLARE cur_cancelled_running_cores_mcpu BIGINT;
  DECLARE cur_n_n_cancelled_creating_jobs INT;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_batch_state FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  SET cur_cancelled = EXISTS (SELECT TRUE
                              FROM batches_cancelled
                              WHERE id = in_batch_id
                              FOR UPDATE);

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM batch_inst_coll_cancellable_resources
    JOIN batches ON batches.id = batch_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON batch_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      batch_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE batch_inst_coll_cancellable_resources.batch_id = in_batch_id AND batch_updates.committed
    GROUP BY user, inst_coll
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_creating_jobs = n_creating_jobs - @n_creating_cancellable_jobs,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs,
      n_cancelled_creating_jobs = n_cancelled_creating_jobs + @n_creating_cancellable_jobs;

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM batch_inst_coll_cancellable_resources WHERE batch_id = in_batch_id;

    INSERT INTO batches_cancelled VALUES (in_batch_id);
  END IF;

  COMMIT;
END $$

DELIMITER ;
