SET foreign_key_checks = 0;

ALTER TABLE jobs MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE jobs ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;

ALTER TABLE batch_attributes MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE batch_attributes ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batch_attributes DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, job_group_id, `key`);

ALTER TABLE batches_cancelled MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE batches_cancelled ADD FOREIGN KEY (`id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batches_cancelled DROP PRIMARY KEY, ADD PRIMARY KEY (id, job_group_id);

ALTER TABLE batches_inst_coll_staging MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE batches_inst_coll_staging ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batches_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `inst_coll`, `token`);

ALTER TABLE batch_inst_coll_cancellable_resources MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE batch_inst_coll_cancellable_resources ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batch_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `job_group_id`, `inst_coll`, `token`);

ALTER TABLE aggregated_batch_resources_v2 MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE aggregated_batch_resources_v2 ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE aggregated_batch_resources_v2 DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `job_group_id`, `resource_id`, `token`);

ALTER TABLE aggregated_batch_resources_v3 MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE aggregated_batch_resources_v3 ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE aggregated_batch_resources_v3 DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `job_group_id`, `resource_id`, `token`);

ALTER TABLE batches_n_jobs_in_complete_states MODIFY COLUMN `job_group_id` INT NOT NULL;
ALTER TABLE batches_n_jobs_in_complete_states MODIFY COLUMN `token` INT NOT NULL;
ALTER TABLE batches_n_jobs_in_complete_states ADD FOREIGN KEY (`id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batches_n_jobs_in_complete_states DROP PRIMARY KEY, ADD PRIMARY KEY (`id`, `job_group_id`, `token`);

SET foreign_key_checks = 1;

DELIMITER $$

DROP TRIGGER IF EXISTS jobs_before_insert $$
CREATE TRIGGER jobs_before_insert BEFORE INSERT ON jobs
FOR EACH ROW
BEGIN
  DECLARE job_group_cancelled BOOLEAN;

  SET job_group_cancelled = EXISTS (SELECT TRUE
                                    FROM batches_cancelled
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
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND aggregated_billing_project_user_resources_v2.migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v3.`usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_parents.parent_id,
      resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs ON attempt_resources.batch_id = jobs.batch_id AND attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_parents ON jobs.batch_id = job_group_parents.batch_id AND jobs.job_group_id = job_group_parents.job_group_id
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id
    ON DUPLICATE KEY UPDATE `usage` = `usage` + msec_diff_rollup * quantity;

    INSERT INTO aggregated_batch_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT attempt_resources.batch_id,
      job_group_parents.parent_id,
      attempt_resources.deduped_resource_id,
      rand_token,
      msec_diff_rollup * quantity
    FROM attempt_resources
    LEFT JOIN jobs ON attempt_resources.batch_id = jobs.batch_id AND attempt_resources.job_id = jobs.job_id
    LEFT JOIN job_group_parents ON jobs.batch_id = job_group_parents.batch_id AND jobs.job_group_id = job_group_parents.job_group_id
    JOIN aggregated_batch_resources_v2 ON
      aggregated_batch_resources_v2.batch_id = attempt_resources.batch_id AND
      aggregated_batch_resources_v2.resource_id = attempt_resources.resource_id AND
      aggregated_batch_resources_v2.token = rand_token
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND migrated = 1
    ON DUPLICATE KEY UPDATE `usage` = aggregated_batch_resources_v3.`usage` + msec_diff_rollup * quantity;

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
    WHERE attempt_resources.batch_id = NEW.batch_id AND attempt_resources.job_id = NEW.job_id AND attempt_resources.attempt_id = NEW.attempt_id AND aggregated_billing_project_user_resources_by_date_v2.migrated = 1
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

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                    FROM batches_cancelled
                                    WHERE id = NEW.batch_id AND job_group_id = NEW.job_group_id
                                    LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    IF NOT (OLD.always_run OR OLD.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      SELECT OLD.batch_id, NEW.update_id, job_group_parents.parent_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    IF NOT OLD.always_run AND (OLD.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs - 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs - 1,
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Running' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      SELECT OLD.batch_id, NEW.update_id, job_group_parents.parent_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs - 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs - 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs - 1,
        running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Creating' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_creating_cancellable_jobs)
      SELECT OLD.batch_id, NEW.update_id, job_group_parents.parent_id, OLD.inst_coll, rand_token, -1
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs - 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_cancelled_creating_jobs - 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, OLD.inst_coll, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs - 1;
    END IF;

  END IF;

  IF NEW.state = 'Ready' THEN
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      SELECT NEW.batch_id, NEW.update_id, job_group_parents.parent_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    IF NOT NEW.always_run AND (NEW.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_ready_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs + 1;
    ELSE
      # runnable
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + 1,
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      SELECT NEW.batch_id, NEW.update_id, job_group_parents.parent_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs + 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_running_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs + 1;
    ELSE
      # running
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs + 1,
        running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Creating' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token, n_creating_cancellable_jobs)
      SELECT NEW.batch_id, NEW.update_id, job_group_parents.parent_id, NEW.inst_coll, rand_token, 1
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = NEW.job_group_id
      ON DUPLICATE KEY UPDATE
        n_creating_cancellable_jobs = n_creating_cancellable_jobs + 1;
    END IF;

    # state = 'Creating' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_cancelled_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_creating_jobs = n_cancelled_creating_jobs + 1;
    ELSE
      # creating
      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_creating_jobs)
      VALUES (cur_user, NEW.inst_coll, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_creating_jobs = n_creating_jobs + 1;
    END IF;
  END IF;
END $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_rollup_time BIGINT;
  DECLARE cur_billing_project VARCHAR(100);
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_job_group_id INT;
  DECLARE msec_diff_rollup BIGINT;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;
  DECLARE cur_billing_date DATE;
  DECLARE bp_user_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE bp_user_resources_by_date_migrated BOOLEAN DEFAULT FALSE;
  DECLARE batch_resources_migrated BOOLEAN DEFAULT FALSE;
  DECLARE job_resources_migrated BOOLEAN DEFAULT FALSE;

  SELECT billing_project, user INTO cur_billing_project, cur_user
  FROM batches WHERE id = NEW.batch_id;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SELECT start_time, rollup_time INTO cur_start_time, cur_rollup_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SELECT job_group_id INTO cur_job_group_id
  FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SET msec_diff_rollup = GREATEST(COALESCE(cur_rollup_time - cur_start_time, 0), 0);

  SET cur_billing_date = CAST(UTC_DATE() AS DATE);

  IF msec_diff_rollup != 0 THEN
    INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, user, resource_id, token, `usage`)
    VALUES (cur_billing_project, cur_user, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO bp_user_resources_migrated
    FROM aggregated_billing_project_user_resources_v2
    WHERE billing_project = cur_billing_project AND user = cur_user AND resource_id = NEW.resource_id AND token = rand_token;

    IF bp_user_resources_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_v3 (billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_batch_resources_v2 (batch_id, job_group_id, resource_id, token, `usage`)
    SELECT NEW.batch_id, parent_id, NEW.resource_id, rand_token, NEW.quantity * msec_diff_rollup
    FROM job_group_parents
    WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = cur_job_group_id
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO batch_resources_migrated
    FROM aggregated_batch_resources_v2
    WHERE batch_id = NEW.batch_id AND resource_id = NEW.resource_id AND token = rand_token;

    IF batch_resources_migrated THEN
      INSERT INTO aggregated_batch_resources_v3 (batch_id, job_group_id, resource_id, token, `usage`)
      SELECT NEW.batch_id, parent_id, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup
      FROM job_group_parents
      WHERE job_group_parents.batch_id = NEW.batch_id AND job_group_parents.job_group_id = cur_job_group_id
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;

    INSERT INTO aggregated_job_resources_v2 (batch_id, job_id, resource_id, `usage`)
    VALUES (NEW.batch_id, NEW.job_id, NEW.resource_id, NEW.quantity * msec_diff_rollup)
    ON DUPLICATE KEY UPDATE
      `usage` = `usage` + NEW.quantity * msec_diff_rollup;

    SELECT migrated INTO job_resources_migrated
    FROM aggregated_job_resources_v2
    WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND resource_id = NEW.resource_id AND token = rand_token;

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
      AND resource_id = NEW.resource_id AND token = rand_token;

    IF bp_user_resources_by_date_migrated THEN
      INSERT INTO aggregated_billing_project_user_resources_by_date_v3 (billing_date, billing_project, user, resource_id, token, `usage`)
      VALUES (cur_billing_date, cur_billing_project, cur_user, NEW.deduped_resource_id, rand_token, NEW.quantity * msec_diff_rollup)
      ON DUPLICATE KEY UPDATE
        `usage` = `usage` + NEW.quantity * msec_diff_rollup;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS recompute_incremental $$
CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_inst_coll_staging;
  DELETE FROM batch_inst_coll_cancellable_resources;
  DELETE FROM user_inst_coll_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_inst_coll_resources` AS (
    SELECT batch_id, job_group_id, batch_state, user, job_inst_coll,
      COALESCE(SUM(1), 0) as n_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancellable), 0) as n_ready_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND cancellable, cores_mcpu, 0)), 0) as ready_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Running' AND cancellable), 0) as n_running_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND cancellable, cores_mcpu, 0)), 0) as running_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND cancellable), 0) as n_creating_cancellable_jobs,
      COALESCE(SUM(job_state = 'Running' AND NOT cancelled), 0) as n_running_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND NOT cancelled, cores_mcpu, 0)), 0) as running_cores_mcpu,
      COALESCE(SUM(job_state = 'Ready' AND runnable), 0) as n_ready_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND runnable, cores_mcpu, 0)), 0) as ready_cores_mcpu,
      COALESCE(SUM(job_state = 'Creating' AND NOT cancelled), 0) as n_creating_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancelled), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(job_state = 'Running' AND cancelled), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(job_state = 'Creating' AND cancelled), 0) as n_cancelled_creating_jobs
    FROM (
      SELECT batches.user,
        batches.id as batch_id,
        jobs.job_group_id as job_group_id,
        batches.state as batch_state,
        jobs.inst_coll as job_inst_coll,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches_cancelled.id IS NOT NULL) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches_cancelled.id IS NOT NULL)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches_cancelled.id IS NOT NULL)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LEFT JOIN batches_cancelled ON jobs.batch_id = batches_cancelled.id AND jobs.job_group_id = batches_cancelled.job_group_id
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, job_group_id, batch_state, user, job_inst_coll
  );

  INSERT INTO batches_inst_coll_staging (batch_id, job_group_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, job_group_id, job_inst_coll, 0,
    CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) AS n_jobs,
    CAST(COALESCE(SUM(n_ready_jobs), 0) AS SIGNED) AS n_ready_jobs,
    CAST(COALESCE(SUM(ready_cores_mcpu), 0) AS SIGNED) AS ready_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  WHERE batch_state = 'open'
  GROUP BY batch_id;

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, job_group_id, inst_coll, token, n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu, n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs)
  SELECT batch_id, job_group_id, job_inst_coll, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu,
    n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs
  FROM tmp_batch_inst_coll_resources;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu, n_creating_jobs,
    n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
  SELECT t.user, t.job_inst_coll, 0, t.n_ready_jobs, t.ready_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu, t.n_creating_jobs,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs, t.n_cancelled_creating_jobs
  FROM (SELECT user, job_inst_coll,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_ready_jobs), 0) as n_ready_jobs,
      COALESCE(SUM(ready_cores_mcpu), 0) as ready_cores_mcpu,
      COALESCE(SUM(n_creating_jobs), 0) as n_creating_jobs,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs,
      COALESCE(SUM(n_cancelled_creating_jobs), 0) as n_cancelled_creating_jobs
    FROM tmp_batch_inst_coll_resources
    WHERE batch_state != 'open'
    GROUP by user, job_inst_coll) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

END $$

DROP PROCEDURE IF EXISTS commit_batch_update $$
CREATE PROCEDURE commit_batch_update(
  IN in_batch_id BIGINT,
  IN in_update_id INT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_update_committed BOOLEAN;
  DECLARE n_jobs_is_correct BOOLEAN;
  DECLARE cur_update_start_job_id INT;
  DECLARE cur_expected_n_jobs INT;
  DECLARE cur_actual_n_jobs INT;

  START TRANSACTION;

  SELECT committed INTO cur_update_committed
  FROM batch_updates
  WHERE batch_id = in_batch_id AND update_id = in_update_id
  FOR UPDATE;

  IF cur_update_committed THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT n_jobs INTO cur_expected_n_jobs
    FROM batch_updates
    WHERE batch_id = in_batch_id AND update_id = in_update_id
    LOCK IN SHARE MODE;

    SELECT CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) INTO cur_actual_n_jobs
    FROM batches_inst_coll_staging
    WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 1;

    # we can only check if overall number of jobs in root job group is correct
    IF cur_expected_n_jobs = cur_actual_n_jobs THEN
      UPDATE batch_updates
      SET committed = 1, time_committed = in_timestamp
      WHERE batch_id = in_batch_id AND update_id = in_update_id;

      UPDATE job_groups
      INNER JOIN (
        SELECT batch_id, job_group_id, CAST(COALESCE(SUM(n_jobs), 0) AS SIGNED) AS staged_n_jobs
        FROM batches_inst_coll_staging
        WHERE batch_id = in_batch_id AND update_id = in_update_id
        GROUP BY batch_id, job_group_id
      ) AS t ON job_groups.batch_id = t.batch_id AND job_groups.job_group_id = t.job_group_id
      SET `state` = 'running', time_completed = NULL, n_jobs = n_jobs + t.staged_n_jobs;

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id AND update_id = in_update_id AND job_group_id = 1
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
              `job_parents`.job_id < cur_update_start_job_id + cur_expected_n_jobs
            GROUP BY `job_parents`.batch_id, `job_parents`.job_id
            FOR UPDATE
          ) AS t
            ON jobs.batch_id = t.batch_id AND
               jobs.job_id = t.job_id
          SET jobs.state = IF(COALESCE(t.n_pending_parents, 0) = 0, 'Ready', 'Pending'),
              jobs.n_pending_parents = COALESCE(t.n_pending_parents, 0),
              jobs.cancelled = IF(COALESCE(t.n_succeeded, 0) = COALESCE(t.n_parents - t.n_pending_parents, 0), jobs.cancelled, 1)
          WHERE jobs.batch_id = in_batch_id AND jobs.job_id >= cur_update_start_job_id AND
              jobs.job_id < cur_update_start_job_id + cur_expected_n_jobs;
      END IF;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;

      SELECT 1 as rc, cur_expected_n_jobs, cur_actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS cancel_job_group $$
CREATE PROCEDURE cancel_job_group(
  IN in_batch_id BIGINT,
  IN in_job_group_id INT
)
BEGIN
  START TRANSACTION;

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
  WHERE batch_inst_coll_cancellable_resources.batch_id = in_batch_id AND
    batch_inst_coll_cancellable_resources.job_group_id = in_job_group_id AND
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

  # subtract cores from all rows that are parents of this job group

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token,
    n_ready_cancellable_jobs, ready_cancellable_cores_mcpu, n_creating_cancellable_jobs, n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  SELECT batch_inst_coll_cancellable_resources.batch_id, batch_inst_coll_cancellable_resources.update_id, job_group_parents.parent_id, inst_coll, 0,
    -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
    -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
    -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
    -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
    -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0))
  FROM batch_inst_coll_cancellable_resources
  INNER JOIN batch_updates ON batch_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
    batch_inst_coll_cancellable_resources.update_id = batch_updates.update_id
  INNER JOIN job_group_parents ON batch_inst_coll_cancellable_resources.batch_id = job_group_parents.batch_id AND
    batch_inst_coll_cancellable_resources.job_group_id = job_group_parents.job_group_id
  WHERE job_group_parents.batch_id = in_batch_id AND
    job_group_parents.job_group_id = in_job_group_id AND
    batch_updates.committed
  GROUP BY batch_inst_coll_cancellable_resources.batch_id, batch_inst_coll_cancellable_resources.update_id,
    job_group_parents.parent_id, inst_coll
  ON DUPLICATE KEY UPDATE
    n_ready_cancellable_jobs = n_ready_cancellable_jobs - @n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - @ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs = n_creating_cancellable_jobs - @n_creating_cancellable_jobs,
    n_running_cancellable_jobs = n_running_cancellable_jobs - @n_running_cancellable_jobs,
    running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - @running_cancellable_cores_mcpu;

  # delete all rows that are children of this job group
  DELETE batch_inst_coll_cancellable_resources FROM batch_inst_coll_cancellable_resources
  LEFT JOIN batch_updates ON batch_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
    batch_inst_coll_cancellable_resources.update_id = batch_updates.update_id
  INNER JOIN job_group_parents ON batch_inst_coll_cancellable_resources.batch_id = job_group_parents.batch_id AND
    batch_inst_coll_cancellable_resources.job_group_id = job_group_parents.job_group_id
  WHERE batch_inst_coll_cancellable_resources.batch_id = in_batch_id AND
    job_group_parents.parent_id = in_job_group_id AND
    batch_updates.committed;

  INSERT INTO batches_cancelled
  SELECT batch_id, job_group_id
  FROM job_group_parents
  WHERE batch_id = in_batch_id AND parent_id = in_job_group_id
  ON DUPLICATE KEY UPDATE job_group_id = batches_cancelled.job_group_id;

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

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id AND batches_cancelled.job_group_id = jobs.job_group_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  SELECT is_pool
  INTO cur_instance_is_pool
  FROM instances
  LEFT JOIN inst_colls ON instances.inst_coll = inst_colls.name
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

DROP PROCEDURE IF EXISTS mark_job_creating $$
CREATE PROCEDURE mark_job_creating(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id AND batches_cancelled.job_group_id = jobs.job_group_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'pending' THEN
    UPDATE jobs SET state = 'Creating', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$

DROP PROCEDURE IF EXISTS mark_job_started $$
CREATE PROCEDURE mark_job_started(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT (jobs.cancelled OR batches_cancelled.id IS NOT NULL) AND NOT jobs.always_run
  INTO cur_job_cancel
  FROM jobs
  LEFT JOIN batches_cancelled ON batches_cancelled.id = jobs.batch_id AND batches_cancelled.job_group_id = jobs.job_group_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  LOCK IN SHARE MODE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
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
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  START TRANSACTION;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

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

    INSERT INTO batches_n_jobs_in_complete_states (id, job_group_id, token, n_completed, n_succeeded, n_failed, n_cancelled)
    SELECT in_batch_id, parent_id, rand_token,
      1,
      (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed'),
      (new_state = 'Error' OR new_state = 'Failed'),
      (new_state = 'Cancelled')
    FROM job_group_parents
    WHERE batch_id = in_batch_id AND job_group_id = cur_job_group_id
    ON DUPLICATE KEY UPDATE n_completed = n_completed + 1,
      n_cancelled = n_cancelled + (new_state = 'Cancelled'),
      n_failed    = n_failed + (new_state = 'Error' OR new_state = 'Failed'),
      n_succeeded = n_succeeded + (new_state != 'Cancelled' AND new_state != 'Error' AND new_state != 'Failed');

    UPDATE jobs
      INNER JOIN `job_parents`
        ON jobs.batch_id = `job_parents`.batch_id AND
           jobs.job_id = `job_parents`.job_id
      SET jobs.state = IF(jobs.n_pending_parents = 1, 'Ready', 'Pending'),
          jobs.n_pending_parents = jobs.n_pending_parents - 1,
          jobs.cancelled = IF(new_state = 'Success', jobs.cancelled, 1)
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

DELIMITER ;
