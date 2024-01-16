DELIMITER $$

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
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    SELECT billing_project, `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
    SELECT batches.billing_project, batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    INNER JOIN aggregated_billing_project_user_resources_v2 ON
      aggregated_billing_project_user_resources_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_v2.user = batches.user AND
      aggregated_billing_project_user_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_group_resources_v2 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_self_and_ancestors.ancestor_id,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs ON attempt_resources.batch_id = jobs.batch_id AND attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_self_and_ancestors ON jobs.batch_id = job_group_self_and_ancestors.batch_id AND jobs.job_group_id = job_group_self_and_ancestors.job_group_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_group_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_self_and_ancestors.ancestor_id,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs ON attempt_resources.batch_id = jobs.batch_id AND attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_self_and_ancestors ON jobs.batch_id = job_group_self_and_ancestors.batch_id AND jobs.job_group_id = job_group_self_and_ancestors.job_group_id
    JOIN aggregated_job_group_resources_v2 ON
      aggregated_job_group_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_job_group_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_job_group_resources_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_group_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    SELECT batch_id, job_id,
      resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
    SELECT attempt_resources.batch_id, attempt_resources.job_id,
      attempt_resources.deduped_resource_id,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN aggregated_job_resources_v2 ON
      aggregated_job_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_job_resources_v2.job_id = attempt_resources.job_id AND
      aggregated_job_resources_v2.resource_id = attempt_resources.resource_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_job_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      billing_project,
      `user`,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
    SELECT cur_billing_date,
      batches.billing_project,
      batches.`user`,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    JOIN batches ON batches.id = attempt_resources.batch_id
    JOIN aggregated_billing_project_user_resources_by_date_v2 ON
      aggregated_billing_project_user_resources_by_date_v2.billing_date = cur_billing_date AND
      aggregated_billing_project_user_resources_by_date_v2.billing_project = batches.billing_project AND
      aggregated_billing_project_user_resources_by_date_v2.user = batches.user AND
      aggregated_billing_project_user_resources_by_date_v2.resource_id = attempt_resources.resource_id AND
      aggregated_billing_project_user_resources_by_date_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v3.`usage` + msec_diff_rollup * quantity;
  END IF;
END $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
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

  SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                    FROM job_groups_cancelled
                                    WHERE id = NEW.batch_id
                                    LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SET always_run = old.always_run; # always_run is immutable
  SET cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

  SET was_marked_cancelled = old.cancelled OR cur_batch_cancelled;
  SET was_cancelled        = NOT always_run AND was_marked_cancelled;
  SET was_cancellable      = NOT always_run AND NOT was_marked_cancelled;

  SET now_marked_cancelled = new.cancelled or cur_batch_cancelled;
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
  DECLARE bp_user_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE bp_user_resources_by_date_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_group_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_resources_migrated BOOLEAN DEFAULT FALSE;

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
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_migrated
    FROM aggregated_billing_project_user_resources_v2
    WHERE billing_project = cur_billing_project AND user = cur_user AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_group_resources_v2 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT NEW.batch_id, ancestor_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup
    FROM job_group_self_and_ancestors
    WHERE job_group_self_and_ancestors.batch_id = NEW.batch_id AND job_group_self_and_ancestors.job_group_id = cur_job_group_id
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_group_resources_migrated
    FROM aggregated_job_group_resources_v2
    WHERE batch_id = NEW.batch_id AND job_group_id = cur_job_group_id AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF job_group_resources_migrated THEN
      INSERT INTO aggregated_job_group_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
      SELECT NEW.batch_id, ancestor_id, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup
      FROM job_group_self_and_ancestors
      WHERE job_group_self_and_ancestors.batch_id = NEW.batch_id AND job_group_self_and_ancestors.job_group_id = cur_job_group_id
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_resources_migrated
    FROM aggregated_job_resources_v2
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND resource_id = NEW.resource_id
    FOR UPDATE;

    IF job_resources_migrated THEN
      INSERT INTO aggregated_job_resources_v3 (batch_id, job_id, resource_id, `usage`)
      VALUES (NEW.batch_id, NEW.job_id, NEW.deduped_resource_id, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_by_date_migrated
    FROM aggregated_billing_project_user_resources_by_date_v2
    WHERE billing_date = cur_billing_date AND billing_project = cur_billing_project AND user = cur_user
      AND resource_id = NEW.resource_id AND token = rand_token
    FOR UPDATE;

    IF bp_user_resources_by_date_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS cancel_job_group $$
CREATE PROCEDURE cancel_job_group(
  IN in_batch_id VARCHAR(100),
  IN in_job_group_id INT
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
                              WHERE id = in_batch_id AND job_group_id = in_job_group_id
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
    DELETE job_group_inst_coll_cancellable_resources FROM job_group_inst_coll_cancellable_resources
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

DELIMITER ;

SET foreign_key_checks = 0;

ALTER TABLE batch_updates ADD COLUMN start_job_group_id INT NOT NULL DEFAULT 1, ALGORITHM=INSTANT;
ALTER TABLE batch_updates ADD COLUMN n_job_groups INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT;
CREATE INDEX `batch_updates_start_job_group_id` ON `batch_updates` (`batch_id`, `start_job_group_id`);

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
