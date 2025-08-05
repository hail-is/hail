CREATE TABLE IF NOT EXISTS `inst_colls` (
  `name` VARCHAR(255) NOT NULL,
  `is_pool` BOOLEAN NOT NULL,
  `boot_disk_size_gb` BIGINT NOT NULL,
  `max_instances` BIGINT NOT NULL,
  `max_live_instances` BIGINT NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;
CREATE INDEX `inst_colls_is_pool` ON `inst_colls` (`is_pool`);

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`)
SELECT 'standard', 1, worker_disk_size_gb, max_instances, pool_size
FROM globals;

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`)
SELECT 'highmem', 1, worker_disk_size_gb, max_instances, pool_size
FROM globals;

INSERT INTO inst_colls (`name`, `is_pool`, `boot_disk_size_gb`, `max_instances`, `max_live_instances`)
SELECT 'highcpu', 1, worker_disk_size_gb, max_instances, pool_size
FROM globals;

CREATE TABLE IF NOT EXISTS `pools` (
  `name` VARCHAR(255) NOT NULL,
  `worker_type` VARCHAR(100) NOT NULL,
  `worker_cores` BIGINT NOT NULL,
  `worker_local_ssd_data_disk` BOOLEAN NOT NULL DEFAULT 1,
  `worker_pd_ssd_data_disk_size_gb` BIGINT NOT NULL DEFAULT 0,
  `enable_standing_worker` BOOLEAN NOT NULL DEFAULT FALSE,
  `standing_worker_cores` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`name`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
SELECT 'standard', 'standard', worker_cores, worker_local_ssd_data_disk,
  worker_pd_ssd_data_disk_size_gb, enable_standing_worker, standing_worker_cores
FROM globals;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
SELECT 'highmem', 'highmem', GREATEST(2, worker_cores), worker_local_ssd_data_disk,
  worker_pd_ssd_data_disk_size_gb, 0, GREATEST(2, standing_worker_cores)
FROM globals;

INSERT INTO pools (`name`, `worker_type`, `worker_cores`, `worker_local_ssd_data_disk`,
  `worker_pd_ssd_data_disk_size_gb`, `enable_standing_worker`, `standing_worker_cores`)
SELECT 'highcpu', 'highcpu', GREATEST(2, worker_cores), worker_local_ssd_data_disk,
  worker_pd_ssd_data_disk_size_gb, 0, GREATEST(2, standing_worker_cores)
FROM globals;

ALTER TABLE instances ADD COLUMN `inst_coll` VARCHAR(255) DEFAULT 'standard';
CREATE INDEX `instances_inst_coll` ON `instances` (`inst_coll`);
CREATE INDEX `instances_removed_inst_coll` ON `instances` (`removed`, `inst_coll`);

ALTER TABLE jobs ADD COLUMN `inst_coll` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE jobs MODIFY COLUMN `inst_coll` VARCHAR(255);
CREATE INDEX `jobs_batch_id_state_always_run_inst_coll_cancelled` ON `jobs` (`batch_id`, `state`, `always_run`, `inst_coll`, `cancelled`);

ALTER TABLE user_resources RENAME user_inst_coll_resources;
ALTER TABLE user_inst_coll_resources ADD COLUMN `inst_coll` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE user_inst_coll_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`user`, inst_coll, token);
ALTER TABLE user_inst_coll_resources ADD FOREIGN KEY (`inst_coll`) REFERENCES inst_colls (`name`) ON DELETE CASCADE;
CREATE INDEX `user_inst_coll_resources_inst_coll` ON `user_inst_coll_resources` (`inst_coll`);

ALTER TABLE batches_staging RENAME batches_inst_coll_staging;
ALTER TABLE batches_inst_coll_staging ADD COLUMN `inst_coll` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE batches_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, inst_coll, token);
ALTER TABLE batches_inst_coll_staging ADD FOREIGN KEY (`inst_coll`) REFERENCES inst_colls (`name`) ON DELETE CASCADE;
CREATE INDEX `batches_inst_coll_staging_inst_coll` ON `batches_inst_coll_staging` (`inst_coll`);

ALTER TABLE batch_cancellable_resources RENAME batch_inst_coll_cancellable_resources;
ALTER TABLE batch_inst_coll_cancellable_resources ADD COLUMN `inst_coll` VARCHAR(255) DEFAULT 'standard';
ALTER TABLE batch_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (batch_id, inst_coll, token);
ALTER TABLE batch_inst_coll_cancellable_resources ADD FOREIGN KEY (`inst_coll`) REFERENCES inst_colls (`name`) ON DELETE CASCADE;
CREATE INDEX `batch_inst_coll_cancellable_resources_inst_coll` ON `batch_inst_coll_cancellable_resources` (`inst_coll`);

ALTER TABLE globals DROP COLUMN `worker_cores`;
ALTER TABLE globals DROP COLUMN `worker_type`;
ALTER TABLE globals DROP COLUMN `worker_disk_size_gb`;
ALTER TABLE globals DROP COLUMN `worker_local_ssd_data_disk`;
ALTER TABLE globals DROP COLUMN `worker_pd_ssd_data_disk_size_gb`;
ALTER TABLE globals DROP COLUMN `standing_worker_cores`;
ALTER TABLE globals DROP COLUMN `max_instances`;
ALTER TABLE globals DROP COLUMN `pool_size`;
ALTER TABLE globals DROP COLUMN `enable_standing_worker`;

DROP TABLE `ready_cores`;

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
  SELECT batch_id, job_inst_coll, 0, n_jobs, n_runnable_jobs, runnable_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  WHERE batch_state = 'open';

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, inst_coll, token, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu)
  SELECT batch_id, job_inst_coll, 0, n_ready_cancellable_jobs, ready_cancellable_cores_mcpu
  FROM tmp_batch_inst_coll_resources
  WHERE NOT batch_cancelled;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu,
    n_running_jobs, running_cores_mcpu,
    n_cancelled_ready_jobs, n_cancelled_running_jobs)
  SELECT t.user, t.job_inst_coll, 0, t.n_runnable_jobs, t.runnable_cores_mcpu,
    t.n_running_jobs, t.running_cores_mcpu,
    t.n_cancelled_ready_jobs, t.n_cancelled_running_jobs
  FROM (SELECT user, job_inst_coll,
      COALESCE(SUM(n_running_jobs), 0) as n_running_jobs,
      COALESCE(SUM(running_cores_mcpu), 0) as running_cores_mcpu,
      COALESCE(SUM(n_runnable_jobs), 0) as n_runnable_jobs,
      COALESCE(SUM(runnable_cores_mcpu), 0) as runnable_cores_mcpu,
      COALESCE(SUM(n_cancelled_ready_jobs), 0) as n_cancelled_ready_jobs,
      COALESCE(SUM(n_cancelled_running_jobs), 0) as n_cancelled_running_jobs
    FROM tmp_batch_inst_coll_resources
    WHERE batch_state != 'open'
    GROUP by user, job_inst_coll) as t;

  DROP TEMPORARY TABLE IF EXISTS `tmp_batch_inst_coll_resources`;

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
    FROM batches_inst_coll_staging
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

      INSERT INTO user_inst_coll_resources (user, inst_coll, token, n_ready_jobs, ready_cores_mcpu)
      SELECT user, inst_coll, 0, @n_ready_jobs := COALESCE(SUM(n_ready_jobs), 0), @ready_cores_mcpu := COALESCE(SUM(ready_cores_mcpu), 0)
      FROM batches_inst_coll_staging
      JOIN batches ON batches.id = batches_inst_coll_staging.batch_id
      WHERE batch_id = in_batch_id
      GROUP BY `user`, inst_coll
      ON DUPLICATE KEY UPDATE
        n_ready_jobs = n_ready_jobs + @n_ready_jobs,
        ready_cores_mcpu = ready_cores_mcpu + @ready_cores_mcpu;

      DELETE FROM batches_inst_coll_staging WHERE batch_id = in_batch_id;

      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, staging_n_jobs as actual_n_jobs, 'wrong number of jobs' as message;
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

  START TRANSACTION;

  SELECT user, `state`, cancelled INTO cur_user, cur_batch_state, cur_cancelled FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_cancelled_ready_jobs, n_cancelled_running_jobs)
    SELECT user, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0), COALESCE(SUM(n_running_cancellable_jobs), 0)
    FROM batch_inst_coll_cancellable_resources
    JOIN batches ON batches.id = batch_inst_coll_cancellable_resources.batch_id
    WHERE batch_id = in_batch_id
    GROUP BY user, inst_coll
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs;

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM batch_inst_coll_cancellable_resources WHERE batch_id = in_batch_id;

    UPDATE batches SET cancelled = 1 WHERE id = in_batch_id;
  END IF;

  COMMIT;
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

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$

DELIMITER ;
