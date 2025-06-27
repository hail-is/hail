ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN `n_creating_cancellable_jobs` INT NOT NULL DEFAULT 0;

ALTER TABLE user_inst_coll_resources ADD COLUMN `n_creating_jobs` INT NOT NULL DEFAULT 0;
ALTER TABLE user_inst_coll_resources ADD COLUMN `n_cancelled_creating_jobs` INT NOT NULL DEFAULT 0;

DELIMITER $$

DROP TRIGGER IF EXISTS jobs_after_update $$
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1, -OLD.cores_mcpu)
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (OLD.batch_id, OLD.inst_coll, rand_token, -1)
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1, NEW.cores_mcpu)
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
      INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_creating_cancellable_jobs)
      VALUES (NEW.batch_id, NEW.inst_coll, rand_token, 1)
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

DROP PROCEDURE IF EXISTS recompute_incremental $$
CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_inst_coll_staging;
  DELETE FROM batch_inst_coll_cancellable_resources;
  DELETE FROM user_inst_coll_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_inst_coll_resources` AS (
    SELECT batch_id, batch_state, batch_cancelled, user, job_inst_coll,
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
        batches.state as batch_state,
        batches.cancelled as batch_cancelled,
        jobs.inst_coll as job_inst_coll,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, batch_state, batch_cancelled, user, job_inst_coll
  );

  INSERT INTO batches_inst_coll_staging (batch_id, inst_coll, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, job_inst_coll, 0, n_jobs, n_ready_jobs, ready_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  WHERE batch_state = 'open';

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu, n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs)
  SELECT batch_id, job_inst_coll, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu,
    n_running_cancellable_jobs, running_cancellable_cores_mcpu, n_creating_cancellable_jobs
  FROM tmp_batch_inst_coll_resources
  WHERE NOT batch_cancelled;

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

DROP PROCEDURE IF EXISTS deactivate_instance $$
CREATE PROCEDURE deactivate_instance(
  IN in_instance_name VARCHAR(100),
  IN in_reason VARCHAR(40),
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE name = in_instance_name FOR UPDATE;

  UPDATE instances
  SET time_deactivated = in_timestamp
  WHERE name = in_instance_name;

  UPDATE attempts
  SET end_time = in_timestamp, reason = in_reason
  WHERE instance_name = in_instance_name;

  IF cur_state = 'pending' or cur_state = 'active' THEN
    UPDATE jobs
    INNER JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
    SET state = 'Ready',
        jobs.attempt_id = NULL
    WHERE instance_name = in_instance_name AND (state = 'Running' OR state = 'Creating');

    UPDATE instances SET state = 'inactive', free_cores_mcpu = cores_mcpu WHERE name = in_instance_name;

    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not live (active or pending)' as message;
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

  SELECT user, `state`, cancelled INTO cur_user, cur_batch_state, cur_cancelled FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

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
    WHERE batch_id = in_batch_id
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

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
  END IF;

  COMMIT;
END $$

DROP PROCEDURE IF EXISTS add_attempt $$
CREATE PROCEDURE add_attempt(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN in_cores_mcpu INT,
  OUT delta_cores_mcpu INT
)
BEGIN
  DECLARE attempt_exists BOOLEAN;
  DECLARE cur_instance_state VARCHAR(40);
  SET delta_cores_mcpu = IFNULL(delta_cores_mcpu, 0);

  SET attempt_exists = EXISTS (SELECT * FROM attempts
                               WHERE batch_id = in_batch_id AND
                                 job_id = in_job_id AND attempt_id = in_attempt_id
                               FOR UPDATE);

  IF NOT attempt_exists AND in_attempt_id IS NOT NULL THEN
    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name) VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name);
    SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name FOR UPDATE;
    # instance pending when attempt is from a job private instance
    IF cur_instance_state = 'pending' OR cur_instance_state = 'active' THEN
      UPDATE instances SET free_cores_mcpu = free_cores_mcpu - in_cores_mcpu WHERE name = in_instance_name;
      SET delta_cores_mcpu = -1 * in_cores_mcpu;
    END IF;
  END IF;
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

  SELECT jobs.state, cores_mcpu, attempt_id,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND batches.`state` = 'running'
    AND job_id = in_job_id
  FOR UPDATE;

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
  FROM jobs WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT end_time INTO cur_end_time FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id
  FOR UPDATE;

  UPDATE attempts
  SET end_time = new_end_time, reason = new_reason
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name FOR UPDATE;

  IF cur_instance_state = 'active' AND cur_end_time IS NULL THEN
    UPDATE instances
    SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
    WHERE name = in_instance_name;

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

  SELECT jobs.state, cores_mcpu,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'pending' THEN
    UPDATE jobs SET state = 'Creating', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
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
      'job state not Ready, Creating, Running or complete' as message;
  END IF;
END $$

DELIMITER ;
