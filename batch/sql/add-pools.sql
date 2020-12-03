CREATE TABLE IF NOT EXISTS `pools` (
  `name` VARCHAR(255) NOT NULL,
  `type` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;

INSERT INTO pools (`name`, `type`) VALUES ('standard', 'standard');

ALTER TABLE instances ADD COLUMN `pool` VARCHAR(255) DEFAULT 'standard';

ALTER TABLE jobs ADD COLUMN `pool` VARCHAR(255) DEFAULT 'standard';
DROP INDEX `jobs_batch_id_state_always_run_cancelled` ON `jobs`;
CREATE INDEX `jobs_batch_id_state_always_run_pool_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `pool`, `cancelled`);

ALTER TABLE user_resources RENAME user_pool_resources;
ALTER TABLE user_pool_resources ADD COLUMN `pool` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE user_pool_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`user`, pool, token);
ALTER TABLE user_pool_resources ADD FOREIGN KEY (`pool`) REFERENCES pools(`name`) ON DELETE CASCADE;
CREATE INDEX `user_pool_resources_pool` ON `user_pool_resources` (`pool`);

ALTER TABLE batches_staging RENAME batches_pool_staging;
ALTER TABLE batches_pool_staging ADD COLUMN `pool` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE batches_pool_staging DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, pool, token);
ALTER TABLE batches_pool_staging ADD FOREIGN KEY (`pool`) REFERENCES pools(`name`) ON DELETE CASCADE;
CREATE INDEX `batches_pool_staging_pool` ON `batches_pool_staging` (`pool`);

ALTER TABLE batch_cancellable_resources RENAME batch_pool_cancellable_resources;
ALTER TABLE batch_pool_cancellable_resources ADD COLUMN `pool` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE batch_pool_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, pool, token);
ALTER TABLE batch_pool_cancellable_resources ADD FOREIGN KEY (`pool`) REFERENCES pools(`name`) ON DELETE CASCADE;
CREATE INDEX `batch_pool_cancellable_resources_pool` ON `batch_pool_cancellable_resources` (`pool`);

DROP TABLE `ready_cores`;

DELIMITER $$

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
      INSERT INTO batch_pool_cancellable_resources (batch_id, pool, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.pool, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs - 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    IF NOT OLD.always_run AND (OLD.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_pool_resources (user, pool, token, n_cancelled_ready_jobs)
      VALUES (cur_user, OLD.pool, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs - 1;
    ELSE
      # runnable
      INSERT INTO user_pool_resources (user, pool, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, OLD.pool, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs - 1,
        ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu;
    END IF;
  ELSEIF OLD.state = 'Running' THEN
    IF NOT (OLD.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_pool_cancellable_resources (batch_id, pool, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (OLD.batch_id, OLD.pool, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs - 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - OLD.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT OLD.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_pool_resources (user, pool, token, n_cancelled_running_jobs)
      VALUES (cur_user, OLD.pool, rand_token, -1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs - 1;
    ELSE
      # running
      INSERT INTO user_pool_resources (user, pool, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, OLD.pool, rand_token, -1, -OLD.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs - 1,
        running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu;
    END IF;
  END IF;

  IF NEW.state = 'Ready' THEN
    IF NOT (NEW.always_run OR NEW.cancelled OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_pool_cancellable_resources (batch_id, pool, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.pool, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_cancellable_jobs = n_ready_cancellable_jobs + 1,
        ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    IF NOT NEW.always_run AND (NEW.cancelled OR cur_batch_cancelled) THEN
      # cancelled
      INSERT INTO user_pool_resources (user, pool, token, n_cancelled_ready_jobs)
      VALUES (cur_user, NEW.pool, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_ready_jobs = n_cancelled_ready_jobs + 1;
    ELSE
      # runnable
      INSERT INTO user_pool_resources (user, pool, token, n_ready_jobs, ready_cores_mcpu)
      VALUES (cur_user, NEW.pool, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + 1,
        ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu;
    END IF;
  ELSEIF NEW.state = 'Running' THEN
    IF NOT (NEW.always_run OR cur_batch_cancelled) THEN
      # cancellable
      INSERT INTO batch_pool_cancellable_resources (batch_id, pool, token, n_running_cancellable_jobs, running_cancellable_cores_mcpu)
      VALUES (NEW.batch_id, NEW.pool, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_cancellable_jobs = n_running_cancellable_jobs + 1,
        running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + NEW.cores_mcpu;
    END IF;

    # state = 'Running' jobs cannot be cancelled at the job level
    IF NOT NEW.always_run AND cur_batch_cancelled THEN
      # cancelled
      INSERT INTO user_pool_resources (user, pool, token, n_cancelled_running_jobs)
      VALUES (cur_user, NEW.pool, rand_token, 1)
      ON DUPLICATE KEY UPDATE
        n_cancelled_running_jobs = n_cancelled_running_jobs + 1;
    ELSE
      # running
      INSERT INTO user_pool_resources (user, pool, token, n_running_jobs, running_cores_mcpu)
      VALUES (cur_user, NEW.pool, rand_token, 1, NEW.cores_mcpu)
      ON DUPLICATE KEY UPDATE
        n_running_jobs = n_running_jobs + 1,
        running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu;
    END IF;
  END IF;
END $$

CREATE PROCEDURE recompute_incremental(
) BEGIN

  DELETE FROM batches_pool_staging;
  DELETE FROM batch_pool_cancellable_resources;
  DELETE FROM user_pool_resources;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_pool_resources`;

  CREATE TEMPORARY TABLE `tmp_batch_pool_resources` AS (
    SELECT batch_id, batch_state, batch_cancelled, user, job_pool,
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
        jobs.pool as job_pool,
        jobs.state as job_state,
        jobs.cores_mcpu,
        NOT (jobs.always_run OR jobs.cancelled OR batches.cancelled) AS cancellable,
        (jobs.always_run OR NOT (jobs.cancelled OR batches.cancelled)) AS runnable,
        (NOT jobs.always_run AND (jobs.cancelled OR batches.cancelled)) AS cancelled
      FROM jobs
      INNER JOIN batches
        ON batches.id = jobs.batch_id
      LOCK IN SHARE MODE) as t
    GROUP BY batch_id, batch_state, batch_cancelled, user, job_pool
  );

  INSERT INTO batches_pool_staging (batch_id, pool, token, n_jobs, n_ready_jobs, ready_cores_mcpu)
  SELECT batch_id, job_pool, 0, n_jobs, n_runnable_jobs, runnable_cores_mcpu
  FROM tmp_batch_pool_resources
  WHERE batch_state = 'open';

  INSERT INTO batch_pool_cancellable_resources (batch_id, pool, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
  SELECT batch_id, job_pool, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu
  FROM tmp_batch_pool_resources
  WHERE NOT batch_cancelled;

  INSERT INTO user_pool_resources (user, pool, token, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu,
    n_cancelled_ready_jobs, n_cancelled_running_jobs)
  SELECT t.user, t.job_pool, 0, t.n_runnable_jobs, t.runnable_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs
  FROM (SELECT user, job_pool,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_runnable_jobs), 0) as n_runnable_jobs,
      COALESCE(SUM(runnable_cores_mcpu), 0) as runnable_cores_mcpu,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs
    FROM tmp_batch_pool_resources
    WHERE batch_state != 'open'
    GROUP by user, job_pool) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_pool_resources`;

END $$

DROP PROCEDURE IF EXISTS close_batch $$
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
    FROM batches_pool_staging
    WHERE batch_id = in_batch_id
    FOR UPDATE;

    SELECT user INTO cur_user FROM batches WHERE id = in_batch_id;

    IF staging_n_jobs = expected_n_jobs THEN
      IF expected_n_jobs = 0 THEN
        UPDATE batches SET `state` = 'complete', time_completed = in_timestamp, time_closed = in_timestamp
          WHERE id = in_batch_id;
      ELSE
        UPDATE batches SET `state` = 'running', time_closed = in_timestamp
          WHERE id = in_batch_id;
      END IF;

      INSERT INTO user_pool_resources (user, pool, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, pool, 0, COALESCE(SUM(staging_n_ready_jobs), 0), COALESCE(SUM(staging_ready_cores_mcpu), 0)
      FROM batches_pool_staging
      JOIN batches ON batches.id = batches_pool_staging.batch_id
      WHERE batch_id = in_batch_id
      GROUP BY user, pool
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + COALESCE(SUM(staging_n_ready_jobs), 0),
        ready_cores_mcpu = ready_cores_mcpu + COALESCE(SUM(staging_ready_cores_mcpu), 0);

      DELETE FROM batches_pool_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  END IF;
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
    INSERT INTO user_pool_resources (user, pool, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_cancelled_ready_jobs, n_cancelled_running_jobs)
    SELECT user, pool, 0,
      -1 * COALESCE(SUM(n_ready_cancellable_jobs), 0), -1 * COALESCE(SUM(ready_cancellable_cores_mcpu), 0),
      -1 * COALESCE(SUM(n_running_cancellable_jobs), 0), -1 * COALESCE(SUM(running_cancellable_cores_mcpu), 0),
      COALESCE(SUM(n_ready_cancellable_jobs), 0), COALESCE(SUM(n_running_cancellable_jobs), 0)
    FROM batch_pool_cancellable_resources
    JOIN batches ON batches.id = batch_pool_cancellable_resources.batch_id
    WHERE batch_id = in_batch_id
    GROUP BY user, pool
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - COALESCE(SUM(n_ready_cancellable_jobs), 0),
      ready_cores_mcpu = ready_cores_mcpu - COALESCE(SUM(ready_cancellable_cores_mcpu), 0),
      n_running_jobs = n_running_jobs - COALESCE(SUM(n_running_cancellable_jobs), 0),
      running_cores_mcpu = running_cores_mcpu - COALESCE(SUM(running_cancellable_cores_mcpu), 0),
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + COALESCE(SUM(n_ready_cancellable_jobs), 0),
      n_cancelled_running_jobs = n_cancelled_running_jobs + COALESCE(SUM(n_running_cancellable_jobs), 0);

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM batch_pool_cancellable_resources WHERE batch_id = in_batch_id;

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
  END IF;

  COMMIT;
END $$

DELIMITER ;
