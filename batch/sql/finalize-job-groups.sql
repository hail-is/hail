DROP TRIGGER IF EXISTS batches_after_update;

SET foreign_key_checks = 0;

# we need to remove the unique index on batch_id, start_job_id because the start_job_id can be repeated if the n_jobs in an update is 0
# `batch_id` was the name of the unique index in my test database
ALTER TABLE batch_updates DROP INDEX `batch_id`, ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE batch_updates ADD COLUMN start_job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
ALTER TABLE batch_updates ADD COLUMN n_job_groups INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX `batch_updates_start_job_group_id` ON `batch_updates` (`batch_id`, `start_job_group_id`);
ALTER TABLE batch_updates DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `start_job_group_id`, `start_job_id`), ALGORITHM=INPLACE, LOCK=NONE;

# the default is NULL for the root job group
ALTER TABLE job_groups ADD COLUMN update_id INT DEFAULT NULL, ALGORITHM=INSTANT;
ALTER TABLE job_groups ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates(batch_id, update_id) ON DELETE CASCADE, ALGORITHM=INPLACE;
CREATE INDEX `job_groups_batch_id_update_id` ON `job_groups` (`batch_id`, `update_id`);

ALTER TABLE jobs MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE jobs ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
CREATE INDEX `jobs_batch_id_ic_state_ar_n_regions_bits_rep_job_group_id` ON `jobs` (`batch_id`, `job_group_id`, `inst_coll`, `state`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);

ALTER TABLE job_group_attributes MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE job_group_attributes ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_group_attributes DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, job_group_id, `key`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE job_groups_cancelled MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE job_groups_cancelled ADD FOREIGN KEY (`id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_groups_cancelled DROP PRIMARY KEY, ADD PRIMARY KEY (id, job_group_id), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE job_groups_inst_coll_staging MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE job_groups_inst_coll_staging ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_groups_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE job_group_inst_coll_cancellable_resources MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE job_group_inst_coll_cancellable_resources ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_group_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE aggregated_job_group_resources_v2 MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE aggregated_job_group_resources_v2 ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE aggregated_job_group_resources_v2 DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `job_group_id`, `resource_id`, `token`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE aggregated_job_group_resources_v3 MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE aggregated_job_group_resources_v3 ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE aggregated_job_group_resources_v3 DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `job_group_id`, `resource_id`, `token`), ALGORITHM=INPLACE, LOCK=NONE;

ALTER TABLE job_groups_n_jobs_in_complete_states MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE job_groups_n_jobs_in_complete_states ADD FOREIGN KEY (`id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE job_groups_n_jobs_in_complete_states DROP PRIMARY KEY, ADD PRIMARY KEY (`id`, `job_group_id`), ALGORITHM=INPLACE, LOCK=NONE;

SET foreign_key_checks = 1;

DELIMITER $$

DROP TRIGGER IF EXISTS jobs_before_insert $$
CREATE TRIGGER jobs_before_insert BEFORE INSERT ON jobs
FOR EACH ROW
BEGIN
  DECLARE job_group_cancelled BOOLEAN;

  SET job_group_cancelled = EXISTS (SELECT TRUE
                                    FROM job_groups_cancelled
                                    WHERE id = NEW.batch_id AND job_group_id = NEW.job_group_id
                                    LOCK IN SHARE MODE);

  IF job_group_cancelled THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = "job group has already been cancelled";
  END IF;
END $$

DROP TRIGGER IF EXISTS attempts_after_update $$
CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT billing_project INTO cur_billing_project FROM batches WHERE id = NEW.batch_id;

  SET msec_diff_rollup = (GREATEST(COALESCE(NEW.rollup_time - NEW.start_time, 0), 0) -
                          GREATEST(COALESCE(OLD.rollup_time - OLD.start_time, 0), 0));

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    SELECT batches.billing_project, batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_group_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_self_and_ancestors.ancestor_id,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs
      ON attempt_resources.batch_id = jobs.batch_id AND
         attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_self_and_ancestors
      ON jobs.batch_id = job_group_self_and_ancestors.batch_id AND
         jobs.job_group_id = job_group_self_and_ancestors.job_group_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_group_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      attempt_resources.deduped_resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      batches.billing_project,
      batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + msec_diff_rollup * quantity;
  END IF;
END $$

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
                                        FROM job_groups_cancelled
                                        WHERE id = NEW.batch_id AND job_group_id = NEW.job_group_id
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

  INSERT INTO job_group_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token,
    n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs,
    n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  SELECT NEW.batch_id, NEW.update_id, job_group_self_and_ancestors.ancestor_id, NEW.inst_coll, rand_token,
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

  INSERT INTO user_inst_coll_resources (user, inst_coll, token,
    n_ready_jobs,
    n_running_jobs,
    n_creating_jobs,
    ready_cores_mcpu,
    running_cores_mcpu,
    n_cancelled_ready_jobs,
    n_cancelled_running_jobs,
    n_cancelled_creating_jobs
  )
  VALUES (cur_user, NEW.inst_coll, rand_token,
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

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_rollup_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_job_group_id INT;
  DECLARE cur_user VARCHAR(100);
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT job_group_id INTO cur_job_group_id
  FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT start_time, rollup_time INTO cur_start_time, cur_rollup_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff_rollup = GREATEST(COALESCE(cur_rollup_time - cur_start_time, 0), 0);

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    INSERT INTO aggregated_job_group_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT NEW.batch_id, ancestor_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup
    FROM job_group_self_and_ancestors
    WHERE job_group_self_and_ancestors.batch_id = NEW.batch_id AND job_group_self_and_ancestors.job_group_id = cur_job_group_id
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.deduped_resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;
  END IF;
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
  DECLARE cur_n_cancelled_ready_jobs INT;
  DECLARE cur_cancelled_ready_cores_mcpu BIGINT;
  DECLARE cur_n_cancelled_running_jobs INT;
  DECLARE cur_cancelled_running_cores_mcpu BIGINT;
  DECLARE cur_n_n_cancelled_creating_jobs INT;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_job_group_state
  FROM job_groups
  WHERE batch_id = in_batch_id AND job_group_id = in_job_group_id
  FOR UPDATE;

  SET cur_cancelled = EXISTS (SELECT TRUE
                              FROM job_groups_cancelled
                              WHERE id = in_batch_id AND job_group_id = in_job_group_id
                              FOR UPDATE);

  IF cur_job_group_state = 'running' AND NOT cur_cancelled THEN
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
    FROM job_group_inst_coll_cancellable_resources
    JOIN batches ON batches.id = job_group_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND
      job_group_inst_coll_cancellable_resources.job_group_id = in_job_group_id AND
      batch_updates.committed
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

    INSERT INTO job_group_inst_coll_cancellable_resources (user, inst_coll, token,
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
    FROM job_group_inst_coll_cancellable_resources
    JOIN batches ON batches.id = job_group_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND
      job_group_inst_coll_cancellable_resources.job_group_id = in_job_group_id AND
      batch_updates.committed
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

    # delete all rows that are children of this job group
    DELETE job_group_inst_coll_cancellable_resources
    FROM job_group_inst_coll_cancellable_resources
    LEFT JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    INNER JOIN job_group_self_and_ancestors ON job_group_inst_coll_cancellable_resources.batch_id = job_group_self_and_ancestors.batch_id AND
      job_group_inst_coll_cancellable_resources.job_group_id = job_group_self_and_ancestors.job_group_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND
      job_group_self_and_ancestors.ancestor_id = in_job_group_id AND
      batch_updates.committed;

    INSERT INTO job_groups_cancelled
    SELECT batch_id, job_group_id
    FROM job_group_self_and_ancestors
    WHERE batch_id = in_batch_id AND ancestor_id = in_job_group_id
    ON DUPLICATE KEY UPDATE job_group_id = job_groups_cancelled.job_group_id;
  END IF;

  COMMIT;
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

      UPDATE job_groups
      INNER JOIN (
        SELECT t.batch_id, t.ancestor_id, CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) AS staged_n_jobs
        FROM job_groups_inst_coll_staging
        INNER JOIN LATERAL (
          SELECT batch_id, ancestor_id
          FROM job_group_self_and_ancestors
          WHERE job_group_self_and_ancestors.batch_id = job_groups_inst_coll_staging.batch_id AND
                job_group_self_and_ancestors.job_group_id = job_groups_inst_coll_staging.job_group_id
        ) AS t ON TRUE
        WHERE job_groups_inst_coll_staging.batch_id = in_batch_id AND job_groups_inst_coll_staging.update_id = in_update_id
        GROUP BY t.batch_id, t.ancestor_id
      ) AS t ON job_groups.batch_id = t.batch_id AND job_groups.job_group_id = t.ancestor_id
      SET `state` = 'running', time_completed = NULL, n_jobs = n_jobs + t.staged_n_jobs;

      # compute global number of new ready jobs from summing all job groups
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0,
        @n_ready_jobs := CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED),
        @ready_cores_mcpu := CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED)
      FROM job_groups_inst_coll_staging
      JOIN batches ON batches.id = job_groups_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM job_groups_inst_coll_staging WHERE batch_id = in_batch_id AND update_id = in_update_id;

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

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 1 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS mark_job_complete $$
CREATE PROCEDURE mark_job_complete(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_state VARCHAR(40),
  IN new_status TEXT,
  IN new_start_time BIGINT,
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40),
  IN new_timestamp BIGINT
)
BEGIN
  DECLARE cur_job_group_id INT;
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;
  DECLARE expected_attempt_id VARCHAR(40);
  DECLARE new_batch_n_completed INT;
  DECLARE total_jobs_in_batch INT;

  START TRANSACTION;

  SELECT n_jobs INTO total_jobs_in_batch FROM batches WHERE id = in_batch_id;

  SELECT state, cores_mcpu, job_group_id
  INTO cur_job_state, cur_cores_mcpu, cur_job_group_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  SELECT end_time INTO cur_end_time FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET start_time = new_start_time, rollup_time = new_end_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;
  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances_free_cores_mcpu
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name;

    SET delta_cores_mcpu = delta_cores_mcpu + cur_cores_mcpu;
  END IF;

  SELECT attempt_id INTO expected_attempt_id FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  IF expected_attempt_id IS NOT NULL AND expected_attempt_id != in_attempt_id THEN
    COMMIT;
    SELECT 2 as rc,
      expected_attempt_id,
      delta_cores_mcpu,
      'input attempt id does not match expected attempt id' as message;
  ELSEIF cur_job_state = 'Ready' OR cur_job_state = 'Creating' OR cur_job_state = 'Running' THEN
    UPDATE jobs
    SET state = new_state, status = new_status, attempt_id = in_attempt_id
    WHERE batch_id = in_batch_id AND job_id = in_job_id;

    SELECT n_completed + 1 INTO new_batch_n_completed
    FROM job_groups_n_jobs_in_complete_states
    WHERE id = in_batch_id AND job_group_id = 0;

    # Grabbing an exclusive lock on batches here could deadlock,
    # but this IF should only execute for the last job
    IF new_batch_n_completed = total_jobs_in_batch THEN
      UPDATE batches
      SET time_completed = new_timestamp,
          `state` = 'complete'
      WHERE id = in_batch_id;
    END IF;

    UPDATE job_groups_n_jobs_in_complete_states
    INNER JOIN (
      SELECT batch_id, ancestor_id
      FROM job_group_self_and_ancestors
      WHERE batch_id = in_batch_id AND job_group_id = cur_job_group_id
      ORDER BY job_group_id ASC
    ) AS t ON job_groups_n_jobs_in_complete_states.id = t.batch_id AND job_groups_n_jobs_in_complete_states.job_group_id = t.ancestor_id
    SET n_completed = n_completed + 1,
        n_cancelled = n_cancelled + (new_state = 'Cancelled'),
        n_failed = n_failed + (new_state = 'Error' OR new_state = 'Failed'),
        n_succeeded = n_succeeded + (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed');

    CALL mark_job_group_complete(in_batch_id, cur_job_group_id, new_timestamp);

    UPDATE jobs
      LEFT JOIN `jobs_telemetry` ON `jobs_telemetry`.batch_id = jobs.batch_id AND `jobs_telemetry`.job_id = jobs.job_id
      INNER JOIN `job_parents`
        ON jobs.batch_id = `job_parents`.batch_id AND
           jobs.job_id = `job_parents`.job_id
      SET jobs.state = IF(jobs.n_pending_parents = 1, 'Ready', 'Pending'),
          jobs.n_pending_parents = jobs.n_pending_parents - 1,
          jobs.cancelled = IF(new_state = 'Success', jobs.cancelled, 1),
          jobs_telemetry.time_ready = IF(jobs.n_pending_parents = 1, new_timestamp, jobs_telemetry.time_ready)
      WHERE jobs.batch_id = in_batch_id AND
            `job_parents`.batch_id = in_batch_id AND
            `job_parents`.parent_id = in_job_id;

    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state,
      delta_cores_mcpu;
  ELSEIF cur_job_state = 'Cancelled' OR cur_job_state = 'Error' OR
         cur_job_state = 'Failed' OR cur_job_state = 'Success' THEN
    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state,
      delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc,
      cur_job_state,
      delta_cores_mcpu,
      'job state not Ready, Creating, Running or complete' as message;
  END IF;
END $$

# https://dev.mysql.com/doc/refman/8.0/en/cursors.html
# https://stackoverflow.com/questions/5817395/how-can-i-loop-through-all-rows-of-a-table-mysql/16350693#16350693
DROP PROCEDURE IF EXISTS mark_job_group_complete $$
CREATE PROCEDURE mark_job_group_complete(
  IN in_batch_id BIGINT,
  IN in_job_group_id INT,
  IN new_timestamp BIGINT
)
BEGIN
  DECLARE cursor_job_group_id INT;
  DECLARE done BOOLEAN DEFAULT FALSE;
  DECLARE total_jobs_in_job_group INT;
  DECLARE cur_n_completed INT;

  DECLARE job_group_cursor CURSOR FOR
  SELECT ancestor_id
  FROM job_group_self_and_ancestors
  WHERE batch_id = in_batch_id AND job_group_id = in_job_group_id
  ORDER BY ancestor_id ASC;

  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

  OPEN job_group_cursor;
  update_job_group_loop: LOOP
    FETCH job_group_cursor INTO cursor_job_group_id;

    IF done THEN
      LEAVE update_job_group_loop;
    END IF;

    SELECT n_jobs INTO total_jobs_in_job_group
    FROM job_groups
    WHERE batch_id = in_batch_id AND job_group_id = cursor_job_group_id
    LOCK IN SHARE MODE;

    SELECT n_completed INTO cur_n_completed
    FROM job_groups_n_jobs_in_complete_states
    WHERE id = in_batch_id AND job_group_id = cursor_job_group_id
    LOCK IN SHARE MODE;

    # Grabbing an exclusive lock on job groups here could deadlock,
    # but this IF should only execute for the last job
    IF cur_n_completed = total_jobs_in_job_group THEN
      UPDATE job_groups
      SET time_completed = new_timestamp,
        `state` = 'complete'
      WHERE batch_id = in_batch_id AND job_group_id = cursor_job_group_id;
    END IF;
  END LOOP;
  CLOSE job_group_cursor;
END $$

DROP PROCEDURE IF EXISTS unschedule_job $$
CREATE PROCEDURE unschedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_attempt_id VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT end_time INTO cur_end_time
  FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET rollup_time = new_end_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances_free_cores_mcpu
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE instances_free_cores_mcpu.name = in_instance_name;

    SET delta_cores_mcpu = cur_cores_mcpu;
  END IF;

  IF (cur_job_state = 'Creating' OR cur_job_state = 'Running') AND cur_attempt_id = in_attempt_id THEN
    UPDATE jobs SET state = 'Ready', attempt_id = NULL WHERE batch_id = in_batch_id AND job_id = in_job_id;
    COMMIT;
    SELECT 0 as rc, delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc, cur_job_state, delta_cores_mcpu,
      'job state not Running or Creating or wrong attempt id' as message;
  END IF;
END $$

DELIMITER ;
