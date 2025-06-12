# All new columns are added in Python
# Indices need to be dropped in a follow-up PR once not being used any longer


ALTER TABLE inst_colls DROP PRIMARY KEY, ADD PRIMARY KEY (`cloud`, `name`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE pools ADD FOREIGN KEY (`cloud`, `name`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE pools DROP PRIMARY KEY, ADD PRIMARY KEY (`cloud`, `name`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE instances ADD FOREIGN KEY (`cloud`, `inst_coll`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
CREATE INDEX `instances_cloud_inst_coll` ON `instances` (`cloud`, `inst_coll`);
CREATE INDEX `instances_removed_cloud_inst_coll` ON `instances` (`removed`, `cloud`, `inst_coll`);
DROP INDEX `instances_inst_coll` ON `instances`;
DROP INDEX `instances_removed_inst_coll` ON `instances`;

ALTER TABLE user_inst_coll_resources ADD FOREIGN KEY (`cloud`, `inst_coll`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE user_inst_coll_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`user`, `cloud`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;
CREATE INDEX `user_inst_coll_resources_cloud_inst_coll` ON `user_inst_coll_resources` (`cloud`, `inst_coll`);
DROP INDEX `user_inst_coll_resources_inst_coll`;

ALTER TABLE job_groups_inst_coll_staging ADD FOREIGN KEY (`cloud`, `inst_coll`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_groups_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `cloud`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;
CREATE INDEX job_groups_inst_coll_staging_cloud_inst_coll ON job_groups_inst_coll_staging (`cloud`, `inst_coll`);
DROP INDEX job_groups_inst_coll_staging_inst_coll;

ALTER TABLE job_group_inst_coll_cancellable_resources ADD FOREIGN KEY (`cloud`, `inst_coll`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_group_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `cloud`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;
CREATE INDEX `job_group_inst_coll_cancellable_resources_cloud_inst_coll` ON `job_group_inst_coll_cancellable_resources` (`cloud`, `inst_coll`);
DROP INDEX `job_group_inst_coll_cancellable_resources_inst_coll`;

ALTER TABLE jobs ADD FOREIGN KEY (`cloud`, `inst_coll`) REFERENCES inst_colls(cloud, name) ON DELETE CASCADE, ALGORITHM=INPLACE;
CREATE INDEX `jobs_batch_id_state_always_run_cloud_inst_coll_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `cloud`, `inst_coll`, `cancelled`);
CREATE INDEX `jobs_batch_id_cloud_ic_state_ar_n_regions_bits_rep_job_id` ON `jobs` (`batch_id`, `cloud`, `inst_coll`, `state`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);
CREATE INDEX `jobs_batch_id_cloud_ic_state_ar_n_regions_bits_rep_job_group_id` ON `jobs` (`batch_id`, `job_group_id`, `cloud`, `inst_coll`, `state`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);
DROP INDEX `jobs_batch_id_state_always_run_inst_coll_cancelled`;
DROP INDEX `jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_id`;
DROP INDEX `jobs_batch_id_cloud_ic_state_ar_n_regions_bits_rep_job_group_id`;

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_job_group_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  DECLARE always_run boolean;
  DECLARE cores_mcpu bigint;

  DECLARE was_marked_cancelled boolean;
  DECLARE was_cancelled        boolean;
  DECLARE was_cancellable      boolean;

  DECLARE now_marked_cancelled boolean;
  DECLARE now_cancelled        boolean;
  DECLARE now_cancellable      boolean;

  DECLARE was_ready boolean;
  DECLARE now_ready boolean;

  DECLARE was_running boolean;
  DECLARE now_running boolean;

  DECLARE was_creating boolean;
  DECLARE now_creating boolean;

  DECLARE delta_n_ready_cancellable_jobs          int;
  DECLARE delta_ready_cancellable_cores_mcpu   bigint;
  DECLARE delta_n_ready_jobs                      int;
  DECLARE delta_ready_cores_mcpu               bigint;
  DECLARE delta_n_cancelled_ready_jobs            int;

  DECLARE delta_n_running_cancellable_jobs        int;
  DECLARE delta_running_cancellable_cores_mcpu bigint;
  DECLARE delta_n_running_jobs                    int;
  DECLARE delta_running_cores_mcpu             bigint;
  DECLARE delta_n_cancelled_running_jobs          int;

  DECLARE delta_n_creating_cancellable_jobs       int;
  DECLARE delta_n_creating_jobs                   int;
  DECLARE delta_n_cancelled_creating_jobs         int;

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SET cur_job_group_cancelled = EXISTS (SELECT TRUE
                                        FROM job_group_self_and_ancestors
                                        INNER JOIN job_groups_cancelled ON job_group_self_and_ancestors.batch_id = job_groups_cancelled.id AND
                                          job_group_self_and_ancestors.ancestor_id = job_groups_cancelled.job_group_id
                                        WHERE batch_id = OLD.batch_id AND job_group_self_and_ancestors.job_group_id = OLD.job_group_id
                                        LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SET always_run = old.always_run; # always_run is immutable
  SET cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

  SET was_marked_cancelled = old.cancelled OR cur_job_group_cancelled;
  SET was_cancelled        = NOT always_run AND was_marked_cancelled;
  SET was_cancellable      = NOT always_run AND NOT was_marked_cancelled;

  SET now_marked_cancelled = new.cancelled or cur_job_group_cancelled;
  SET now_cancelled        = NOT always_run AND now_marked_cancelled;
  SET now_cancellable      = NOT always_run AND NOT now_marked_cancelled;

  # NB: was_cancelled => now_cancelled b/c you cannot be uncancelled

  SET was_ready    = old.state = 'Ready';
  SET now_ready    = new.state = 'Ready';
  SET was_running  = old.state = 'Running';
  SET now_running  = new.state = 'Running';
  SET was_creating = old.state = 'Creating';
  SET now_creating = new.state = 'Creating';

  SET delta_n_ready_cancellable_jobs        = (-1 * was_ready    *  was_cancellable  )     + (now_ready    *  now_cancellable  ) ;
  SET delta_n_ready_jobs                    = (-1 * was_ready    * (NOT was_cancelled))    + (now_ready    * (NOT now_cancelled));
  SET delta_n_cancelled_ready_jobs          = (-1 * was_ready    *  was_cancelled    )     + (now_ready    *  now_cancelled    ) ;

  SET delta_n_running_cancellable_jobs      = (-1 * was_running  *  was_cancellable  )     + (now_running  *  now_cancellable  ) ;
  SET delta_n_running_jobs                  = (-1 * was_running  * (NOT was_cancelled))    + (now_running  * (NOT now_cancelled));
  SET delta_n_cancelled_running_jobs        = (-1 * was_running  *  was_cancelled    )     + (now_running  *  now_cancelled    ) ;

  SET delta_n_creating_cancellable_jobs     = (-1 * was_creating *  was_cancellable  )     + (now_creating *  now_cancellable  ) ;
  SET delta_n_creating_jobs                 = (-1 * was_creating * (NOT was_cancelled))    + (now_creating * (NOT now_cancelled));
  SET delta_n_cancelled_creating_jobs       = (-1 * was_creating *  was_cancelled    )     + (now_creating *  now_cancelled    ) ;

  SET delta_ready_cancellable_cores_mcpu    = delta_n_ready_cancellable_jobs * cores_mcpu;
  SET delta_ready_cores_mcpu                = delta_n_ready_jobs * cores_mcpu;

  SET delta_running_cancellable_cores_mcpu  = delta_n_running_cancellable_jobs * cores_mcpu;
  SET delta_running_cores_mcpu              = delta_n_running_jobs * cores_mcpu;

  INSERT INTO job_group_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, cloud, inst_coll, token,
    n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs,
    n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  SELECT NEW.batch_id, NEW.update_id, job_group_self_and_ancestors.ancestor_id, NEW.cloud, NEW.inst_coll, rand_token,
    delta_n_ready_cancellable_jobs,
    delta_ready_cancellable_cores_mcpu,
    delta_n_creating_cancellable_jobs,
    delta_n_running_cancellable_jobs,
    delta_running_cancellable_cores_mcpu
  FROM job_group_self_and_ancestors
  WHERE job_group_self_and_ancestors.batch_id = NEW.batch_id AND job_group_self_and_ancestors.job_group_id = NEW.job_group_id
  ON DUPLICATE KEY UPDATE
    n_ready_cancellable_jobs = n_ready_cancellable_jobs + delta_n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + delta_ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs = n_creating_cancellable_jobs + delta_n_creating_cancellable_jobs,
    n_running_cancellable_jobs = n_running_cancellable_jobs + delta_n_running_cancellable_jobs,
    running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + delta_running_cancellable_cores_mcpu;

  INSERT INTO user_inst_coll_resources (user, cloud, inst_coll, token,
    n_ready_jobs,
    n_running_jobs,
    n_creating_jobs,
    ready_cores_mcpu,
    running_cores_mcpu,
    n_cancelled_ready_jobs,
    n_cancelled_running_jobs,
    n_cancelled_creating_jobs
  )
  VALUES (cur_user, NEW.cloud, NEW.inst_coll, rand_token,
    delta_n_ready_jobs,
    delta_n_running_jobs,
    delta_n_creating_jobs,
    delta_ready_cores_mcpu,
    delta_running_cores_mcpu,
    delta_n_cancelled_ready_jobs,
    delta_n_cancelled_running_jobs,
    delta_n_cancelled_creating_jobs
  )
  ON DUPLICATE KEY UPDATE
    n_ready_jobs = n_ready_jobs + delta_n_ready_jobs,
    n_running_jobs = n_running_jobs + delta_n_running_jobs,
    n_creating_jobs = n_creating_jobs + delta_n_creating_jobs,
    ready_cores_mcpu = ready_cores_mcpu + delta_ready_cores_mcpu,
    running_cores_mcpu = running_cores_mcpu + delta_running_cores_mcpu,
    n_cancelled_ready_jobs = n_cancelled_ready_jobs + delta_n_cancelled_ready_jobs,
    n_cancelled_running_jobs = n_cancelled_running_jobs + delta_n_cancelled_running_jobs,
    n_cancelled_creating_jobs = n_cancelled_creating_jobs + delta_n_cancelled_creating_jobs;
END $$

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
    SELECT CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) INTO staging_n_jobs
    FROM job_groups_inst_coll_staging
    WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 0
    FOR UPDATE;

    IF staging_n_jobs = expected_n_jobs THEN
      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE batch_id = in_batch_id AND update_id = in_update_id;

      IF expected_n_jobs > 0 THEN
        UPDATE batches SET
          `state` = 'running',
          time_completed = NULL,
          n_jobs = n_jobs + expected_n_jobs
        WHERE id = in_batch_id;

        UPDATE job_groups
        INNER JOIN (
          SELECT batch_id, job_group_id, CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) AS staged_n_jobs
          FROM job_groups_inst_coll_staging
          WHERE batch_id = in_batch_id AND update_id = in_update_id
          GROUP BY batch_id, job_group_id
        ) AS t ON job_groups.batch_id = t.batch_id AND job_groups.job_group_id = t.job_group_id
        SET `state` = IF(t.staged_n_jobs > 0, 'running', job_groups.state), time_completed = NULL, n_jobs = n_jobs + t.staged_n_jobs;

        # compute global number of new ready jobs from taking value from root job group only
        INSERT INTO user_inst_coll_resources (user, cloud, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
        SELECT user, cloud, inst_coll, 0,
          @n_ready_jobs := CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED),
          @ready_cores_mcpu := CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED)
        FROM job_groups_inst_coll_staging
        JOIN batches ON batches.id = job_groups_inst_coll_staging.batch_id
        WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 0
        GROUP BY `user`, cloud, inst_coll
        ON DUPLICATE KEY UPDATE
          n_ready_jobs = n_ready_jobs + @n_ready_jobs,
          ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

        # Committing a batch update, like any operation, must be O(1) time. The number of descendant groups is unbounded,
        # so we do not delete rows from job_groups_inst_coll_staging. Instead, the deletion of rows is handled by main.py.

        IF in_update_id != 1 THEN
          SELECT start_job_id INTO cur_update_start_job_id FROM batch_updates WHERE batch_id = in_batch_id AND update_id = in_update_id;

          UPDATE jobs
            LEFT JOIN `jobs_telemetry` ON `jobs_telemetry`.batch_id = jobs.batch_id AND `jobs_telemetry`.job_id = jobs.job_id
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
                jobs.cancelled = IF(COALESCE(t.n_succeeded, 0) = COALESCE(t.n_parents - t.n_pending_parents, 0), jobs.cancelled, 1),
                jobs_telemetry.time_ready = IF(COALESCE(t.n_pending_parents, 0) = 0 AND jobs_telemetry.time_ready IS NULL, in_timestamp, jobs_telemetry.time_ready)
            WHERE jobs.batch_id = in_batch_id AND jobs.job_id >= cur_update_start_job_id AND
                jobs.job_id < cur_update_start_job_id + staging_n_jobs;
        END IF;
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
                              FROM job_groups_cancelled
                              WHERE id = in_batch_id
                              FOR UPDATE);

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, cloud, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, cloud, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM job_group_inst_coll_cancellable_resources
    JOIN batches ON batches.id = job_group_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND batch_updates.committed
    GROUP BY user, cloud, inst_coll
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
    DELETE FROM job_group_inst_coll_cancellable_resources WHERE batch_id = in_batch_id;

    # cancel root job group only
    INSERT INTO job_groups_cancelled (id, job_group_id) VALUES (in_batch_id, 0);
  END IF;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS cancel_job_group $$
CREATE PROCEDURE cancel_job_group(
  IN in_batch_id VARCHAR(100),
  IN in_job_group_id INT
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_job_group_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_job_group_state
  FROM job_groups
  WHERE batch_id = in_batch_id AND job_group_id = in_job_group_id
  FOR UPDATE;

  SET cur_cancelled = EXISTS (SELECT TRUE
                              FROM job_group_self_and_ancestors
                              INNER JOIN job_groups_cancelled ON job_group_self_and_ancestors.batch_id = job_groups_cancelled.id AND
                                job_group_self_and_ancestors.ancestor_id = job_groups_cancelled.job_group_id
                              WHERE batch_id = in_batch_id AND job_group_self_and_ancestors.job_group_id = in_job_group_id
                              FOR UPDATE);

  IF NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, cloud, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, cloud, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM job_group_inst_coll_cancellable_resources
    INNER JOIN batches ON job_group_inst_coll_cancellable_resources.batch_id = batches.id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND
      job_group_inst_coll_cancellable_resources.job_group_id = in_job_group_id AND
      batch_updates.committed
    GROUP BY user, cloud, inst_coll
    FOR UPDATE
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_creating_jobs = n_creating_jobs - @n_creating_cancellable_jobs,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs,
      n_cancelled_creating_jobs = n_cancelled_creating_jobs + @n_creating_cancellable_jobs;

    INSERT INTO job_group_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, cloud, inst_coll, token,
      n_ready_cancellable_jobs,
      ready_cancellable_cores_mcpu,
      n_creating_cancellable_jobs,
      n_running_cancellable_jobs,
      running_cancellable_cores_mcpu)
    SELECT batch_id, update_id, ancestor_id, cloud, inst_coll, 0,
      -1 * (@jg_n_ready_cancellable_jobs := old_n_ready_cancellable_jobs),
      -1 * (@jg_ready_cancellable_cores_mcpu := old_ready_cancellable_cores_mcpu),
      -1 * (@jg_n_creating_cancellable_jobs := old_n_creating_cancellable_jobs),
      -1 * (@jg_n_running_cancellable_jobs := old_n_running_cancellable_jobs),
      -1 * (@jg_running_cancellable_cores_mcpu := old_running_cancellable_cores_mcpu)
    FROM job_group_self_and_ancestors
    INNER JOIN LATERAL (
      SELECT update_id, cloud, inst_coll, COALESCE(SUM(n_ready_cancellable_jobs), 0) AS old_n_ready_cancellable_jobs,
        COALESCE(SUM(ready_cancellable_cores_mcpu), 0) AS old_ready_cancellable_cores_mcpu,
        COALESCE(SUM(n_creating_cancellable_jobs), 0) AS old_n_creating_cancellable_jobs,
        COALESCE(SUM(n_running_cancellable_jobs), 0) AS old_n_running_cancellable_jobs,
        COALESCE(SUM(running_cancellable_cores_mcpu), 0) AS old_running_cancellable_cores_mcpu
      FROM job_group_inst_coll_cancellable_resources
      WHERE job_group_inst_coll_cancellable_resources.batch_id = job_group_self_and_ancestors.batch_id AND
        job_group_inst_coll_cancellable_resources.job_group_id = job_group_self_and_ancestors.job_group_id
      GROUP BY update_id, cloud, inst_coll
      FOR UPDATE
    ) AS t ON TRUE
    WHERE job_group_self_and_ancestors.batch_id = in_batch_id AND job_group_self_and_ancestors.job_group_id = in_job_group_id
    ON DUPLICATE KEY UPDATE
      n_ready_cancellable_jobs = n_ready_cancellable_jobs - @jg_n_ready_cancellable_jobs,
      ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - @jg_ready_cancellable_cores_mcpu,
      n_creating_cancellable_jobs = n_creating_cancellable_jobs - @jg_n_creating_cancellable_jobs,
      n_running_cancellable_jobs = n_running_cancellable_jobs - @jg_n_running_cancellable_jobs,
      running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - @jg_running_cancellable_cores_mcpu;

    # Group cancellation, like any operation, must be O(1) time. The number of descendant groups is unbounded,
    # so we neither delete rows from job_group_inst_coll_cancellable_resources nor update job_groups_cancelled.
    # The former is handled by main.py. In the latter case, group cancellation state is implicitly defined by an
    # upwards traversal on the ancestor tree.

    INSERT INTO job_groups_cancelled (id, job_group_id)
    VALUES (in_batch_id, in_job_group_id);
  END IF;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS schedule_job $$
CREATE PROCEDURE schedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_attempt_id VARCHAR(40);
  DECLARE delta_cores_mcpu INT;
  DECLARE cur_instance_is_pool BOOLEAN;

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR job_groups_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN job_groups_cancelled ON job_groups_cancelled.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  SELECT is_pool
  INTO cur_instance_is_pool
  FROM instances
  LEFT JOIN inst_colls ON instances.cloud = inst_cols.cloud AND instances.inst_coll = inst_colls.name
  WHERE instances.name = in_instance_name;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  IF cur_instance_is_pool THEN
    IF delta_cores_mcpu = 0 THEN
      SET delta_cores_mcpu = cur_cores_mcpu;
    ELSE
      SET delta_cores_mcpu = 0;
    END IF;
  END IF;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF (cur_job_state = 'Ready' OR cur_job_state = 'Creating') AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
    COMMIT;
    SELECT 0 as rc, in_instance_name, delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc,
      cur_job_state,
      cur_job_cancel,
      cur_instance_state,
      in_instance_name,
      cur_attempt_id,
      delta_cores_mcpu,
      'job not Ready or cancelled or instance not active, but attempt already exists' as message;
  END IF;
END $$


