
ALTER TABLE `batches` ADD COLUMN `state` VARCHAR(40);

UPDATE `batches`
SET `state` = CASE
  WHEN NOT closed THEN 'open'
  WHEN n_completed = n_jobs THEN 'complete'
  ELSE 'running'
  END;

ALTER TABLE `batches`
  MODIFY COLUMN `state` VARCHAR(40) NOT NULL,
  DROP COLUMN closed;

DROP INDEX `batches_user` ON `batches`;
CREATE INDEX `batches_user_state_cancelled` ON `batches` (`user`, `state`, `cancelled`);

DROP INDEX `jobs_state` ON `jobs`;
CREATE INDEX `jobs_batch_id_state_always_run_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `cancelled`);

ALTER TABLE `user_resources`
  ADD COLUMN `n_cancelled_ready_jobs` INT NOT NULL DEFAULT 0,
  ADD COLUMN `n_cancelled_running_jobs` INT NOT NULL DEFAULT 0;

CREATE TABLE `batch_cancellable_resources` (
  `batch_id` BIGINT NOT NULL,
  `token` INT NOT NULL,
  # neither run_always nor cancelled
  `n_ready_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `ready_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  `n_running_cancellable_jobs` INT NOT NULL DEFAULT 0,
  `running_cancellable_cores_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_staging;
  DELETE FROM batch_cancellable_resources;
  DELETE FROM ready_cores;
  DELETE FROM user_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_resources` AS (
    SELECT batch_id, batch_state, batch_cancelled, user,
      COALESCE(SUM(1), 0) as n_jobs,
      COALESCE(SUM(job_state = 'Ready' AND cancellable), 0) as n_ready_cancellable_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND cancellable, cores_mcpu, 0)), 0) as ready_cancellable_cores_mcpu,
      COALESCE(SUM(job_state = 'Running' AND NOT cancelled), 0) as n_running_jobs,
      COALESCE(SUM(IF(job_state = 'Running' AND NOT cancelled, cores_mcpu, 0)), 0) as running_cores_mcpu,
      COALESCE(SUM(job_state = 'Ready' AND runnable), 0) as n_runnable_jobs,
      COALESCE(SUM(IF(job_state = 'Ready' AND runnable, cores_mcpu, 0)), 0) as runnable_cores_mcpu,
      COALESCE(SUM(job_state = 'Ready' AND cancelled), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(job_state = 'Running' AND cancelled), 0) as n_cancelled_running_jobs
    FROM (
      SELECT batches.user,
        batches.id as batch_id,
        batches.state as batch_state,
        batches.cancelled as batch_cancelled,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, batch_state, batch_cancelled, user
  );

  INSERT INTO batches_staging (batch_id, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, 0, n_jobs, n_runnable_jobs, runnable_cores_mcpu
  FROM tmp_batch_resources
  WHERE batch_state = 'open';

  INSERT INTO batch_cancellable_resources (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
  SELECT batch_id, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu
  FROM tmp_batch_resources
  WHERE NOT batch_cancelled;

  INSERT INTO ready_cores (token, ready_cores_mcpu)
  SELECT 0, t.runnable_cores_mcpu
  FROM (SELECT COALESCE(SUM(runnable_cores_mcpu), 0) as runnable_cores_mcpu
    FROM tmp_batch_resources
    WHERE batch_state != 'open') as t;

  INSERT INTO user_resources (token, user, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu,
    n_cancelled_ready_jobs, n_cancelled_running_jobs)
  SELECT 0, t.user, t.n_runnable_jobs, t.runnable_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs
  FROM (SELECT user,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_runnable_jobs), 0) as n_runnable_jobs,
      COALESCE(SUM(runnable_cores_mcpu), 0) as runnable_cores_mcpu,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs
    FROM tmp_batch_resources
    WHERE batch_state != 'open'
    GROUP by user) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_resources`;

END $$

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

  START TRANSACTION;

  SELECT user, `state`, cancelled INTO cur_user, cur_batch_state, cur_cancelled FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    SELECT COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(ready_cancellable_cores_mcpu), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(running_cancellable_cores_mcpu), 0)
    INTO cur_n_cancelled_ready_jobs, cur_cancelled_ready_cores_mcpu,
      cur_n_cancelled_running_jobs, cur_cancelled_running_cores_mcpu
    FROM batch_cancellable_resources
    WHERE batch_id = in_batch_id
    LOCK IN SHARE MODE;

    INSERT INTO user_resources (user, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_cancelled_ready_jobs, n_cancelled_running_jobs)
    VALUES (cur_user, 0,
      -cur_n_cancelled_ready_jobs, -cur_cancelled_ready_cores_mcpu,
      -cur_n_cancelled_running_jobs, -cur_cancelled_running_cores_mcpu,
      cur_n_cancelled_ready_jobs, cur_n_cancelled_running_jobs)
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - cur_n_cancelled_ready_jobs,
      ready_cores_mcpu = ready_cores_mcpu - cur_cancelled_ready_cores_mcpu,
      n_running_jobs = n_running_jobs - cur_n_cancelled_running_jobs,
      running_cores_mcpu = running_cores_mcpu - cur_cancelled_running_cores_mcpu,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + cur_n_cancelled_ready_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + cur_n_cancelled_running_jobs;

    INSERT INTO ready_cores (token, ready_cores_mcpu)
    VALUES (0, -cur_cancelled_ready_cores_mcpu)
    ON DUPLICATE KEY UPDATE
      ready_cores_mcpu = ready_cores_mcpu - cur_cancelled_ready_cores_mcpu;

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM batch_cancellable_resources WHERE batch_id = in_batch_id;

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
  END IF;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS close_batch;
CREATE PROCEDURE close_batch(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE expected_n_jobs INT;
  DECLARE staging_n_jobs INT;
  DECLARE staging_n_ready_jobs INT;
  DECLARE staging_ready_cores_mcpu BIGINT;
  DECLARE cur_user VARCHAR(100);

  START TRANSACTION;

  SELECT `state`, n_jobs INTO cur_batch_state, expected_n_jobs FROM batches
  WHERE id = in_batch_id AND NOT deleted
  FOR UPDATE;

  IF cur_batch_state != 'open' THEN
    COMMIT;
    SELECT 0 as rc;
  ELSE
    SELECT COALESCE(SUM(n_jobs), 0), COALESCE(SUM(n_ready_jobs), 0), COALESCE(SUM(ready_cores_mcpu), 0)
    INTO staging_n_jobs, staging_n_ready_jobs, staging_ready_cores_mcpu
    FROM batches_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      IF expected_n_jobs = 0 THEN
        UPDATE batches SET `state` = 'complete', time_completed = in_timestamp
          WHERE id = in_batch_id;
      ELSE
        UPDATE batches SET `state` = 'running'
          WHERE id = in_batch_id;
      END IF;

      INSERT INTO ready_cores (token, ready_cores_mcpu)
      VALUES (0, staging_ready_cores_mcpu)
      ON DUPLICATE KEY UPDATE ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu;

      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, 0, staging_n_ready_jobs, staging_ready_cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + staging_n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + staging_ready_cores_mcpu;

      DELETE FROM batches_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
END $$

DROP PROCEDURE IF EXISTS schedule_job;
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

  START TRANSACTION;

  SELECT jobs.state, cores_mcpu, attempt_id,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND batches.`state` = 'running'
    AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  IF delta_cores_mcpu = 0 THEN
    SET delta_cores_mcpu = cur_cores_mcpu;
  ELSE
    SET delta_cores_mcpu = 0;
  END IF;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
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

DROP PROCEDURE IF EXISTS mark_job_started;
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

  SELECT jobs.state, cores_mcpu,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND batches.`state` = 'running'
    AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$

DROP PROCEDURE IF EXISTS mark_job_complete;
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
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_end_time BIGINT;
  DECLARE delta_cores_mcpu INT DEFAULT 0;
  DECLARE expected_attempt_id VARCHAR(40);

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  SELECT end_time INTO cur_end_time FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET start_time = new_start_time, end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name FOR UPDATE;
  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE name = in_instance_name;

    SET delta_cores_mcpu = delta_cores_mcpu + cur_cores_mcpu;
  END IF;

  SELECT attempt_id INTO expected_attempt_id FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  IF expected_attempt_id != in_attempt_id THEN
    COMMIT;
    SELECT 2 as rc,
      expected_attempt_id,
      delta_cores_mcpu,
      'input attempt id does not match expected attempt id' as message;
  ELSEIF cur_job_state = 'Ready' OR cur_job_state = 'Running' THEN
    UPDATE jobs
    SET state = new_state, status = new_status, attempt_id = NULL
    WHERE batch_id = in_batch_id AND job_id = in_job_id;

    UPDATE batches SET n_completed = n_completed + 1 WHERE id = in_batch_id;
    UPDATE batches
      SET time_completed = new_timestamp,
          `state` = 'complete'
      WHERE id = in_batch_id AND n_completed = batches.n_jobs;

    IF new_state = 'Cancelled' THEN
      UPDATE batches SET n_cancelled = n_cancelled + 1 WHERE id = in_batch_id;
    ELSEIF new_state = 'Error' OR new_state = 'Failed' THEN
      UPDATE batches SET n_failed = n_failed + 1 WHERE id = in_batch_id;
    ELSE
      UPDATE batches SET n_succeeded = n_succeeded + 1 WHERE id = in_batch_id;
    END IF;

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
      'job state not Ready, Running or complete' as message;
  END IF;
END $$

DROP TRIGGER IF EXISTS jobs_after_update;
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  SELECT user, cancelled INTO cur_user, cur_batch_cancelled FROM batches
  WHERE id = NEW.batch_id
  LOCK IN SHARE MODE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  IF OLD.state = 'Ready' THEN
    IF NOT (OLD.always_run OR OLD.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_cancellable_resources (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    IF NOT OLD.always_run AND (OLD.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_resources (user, token, n_cancelled_ready_jobs)
      VALUES (cur_user, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs - 1;
    ELSE
      # runnable
      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs - 1,
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;

      INSERT INTO ready_cores (token, ready_cores_mcpu)
      VALUES (rand_token, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Running' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_cancellable_resources (batch_id, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs - 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_resources (user, token, n_cancelled_running_jobs)
      VALUES (cur_user, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs - 1;
    ELSE
      # running
      INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs - 1,
        running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
    END IF;
  END IF;

  IF NEW.state = 'Ready' THEN
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_cancellable_resources (batch_id, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    IF NOT NEW.always_run AND (NEW.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_resources (user, token, n_cancelled_ready_jobs)
      VALUES (cur_user, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs + 1;
    ELSE
      # runnable
      INSERT INTO user_resources (user, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + 1,
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;

      INSERT INTO ready_cores (token, ready_cores_mcpu)
      VALUES (rand_token, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_cancellable_resources (batch_id, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs + 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_resources (user, token, n_cancelled_running_jobs)
      VALUES (cur_user, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs + 1;
    ELSE
      # running
      INSERT INTO user_resources (user, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs + 1,
        running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
    END IF;
  END IF;
END $$

DELIMITER ;

CALL recompute_incremental();
