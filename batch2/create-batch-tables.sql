CREATE TABLE IF NOT EXISTS `globals` (
  `instance_id` VARCHAR(100) NOT NULL,
  `internal_token` VARCHAR(100) NOT NULL,
  `worker_cores` BIGINT NOT NULL,
  `worker_type` VARCHAR(100) NOT NULL,
  `worker_disk_size_gb` BIGINT NOT NULL,
  `max_instances` BIGINT NOT NULL,
  `pool_size` BIGINT NOT NULL
) ENGINE = InnoDB;


CREATE TABLE IF NOT EXISTS `billing_projects` (
  `name` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `billing_project_users` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`billing_project`, `user`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `instances` (
  `name` VARCHAR(100) NOT NULL,
  `state` VARCHAR(40) NOT NULL,
  `activation_token` VARCHAR(100),
  `token` VARCHAR(100) NOT NULL,
  `cores_mcpu` INT NOT NULL,
  `free_cores_mcpu` INT NOT NULL,
  `time_created` BIGINT NOT NULL,
  `failed_request_count` INT NOT NULL DEFAULT 0,
  `last_updated` BIGINT NOT NULL,
  `ip_address` VARCHAR(100),
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `user_resources` (
  `user` VARCHAR(100) NOT NULL,
  `n_ready_jobs` INT NOT NULL DEFAULT 0,
  `n_running_jobs` INT NOT NULL DEFAULT 0,
  `ready_cores_mcpu` INT NOT NULL DEFAULT 0,
  `running_cores_mcpu` INT NOT NULL DEFAULT 0,
  PRIMARY KEY (`user`)
) ENGINE = InnoDB;
CREATE INDEX `user_resources_ready_cores_mcpu` ON `user_resources` (`user`);

CREATE TABLE IF NOT EXISTS `batches` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `userdata` VARCHAR(65535) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `attributes` VARCHAR(65535),
  `callback` VARCHAR(65535),
  `deleted` BOOLEAN NOT NULL DEFAULT FALSE,
  `cancelled` BOOLEAN NOT NULL DEFAULT FALSE,
  `closed` BOOLEAN NOT NULL DEFAULT FALSE,
  `n_jobs` INT NOT NULL,
  `n_completed` INT NOT NULL DEFAULT 0,
  `n_succeeded` INT NOT NULL DEFAULT 0,
  `n_failed` INT NOT NULL DEFAULT 0,
  `n_cancelled` INT NOT NULL DEFAULT 0,
  `time_created` BIGINT NOT NULL,
  `time_completed` BIGINT,
  `msec_mcpu` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`user`) REFERENCES user_resources(user) ON DELETE CASCADE,
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name)
) ENGINE = InnoDB;
CREATE INDEX `batches_user` ON `batches` (`user`);
CREATE INDEX `batches_deleted` ON `batches` (`deleted`);

CREATE TABLE IF NOT EXISTS `jobs` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `state` VARCHAR(40) NOT NULL,
  `spec` VARCHAR(65535) NOT NULL,
  `always_run` BOOLEAN NOT NULL,
  `cores_mcpu` INT NOT NULL,
  `status` VARCHAR(65535),
  `n_pending_parents` INT NOT NULL,
  `cancelled` BOOLEAN NOT NULL DEFAULT FALSE,
  `msec_mcpu` BIGINT NOT NULL DEFAULT 0,
  `attempt_id` VARCHAR(40),
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `jobs_state` ON `jobs` (`state`);

CREATE TABLE IF NOT EXISTS `attempts` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `attempt_id` VARCHAR(40) NOT NULL,
  `instance_name` VARCHAR(100),  
  `start_time` BIGINT,
  `end_time` BIGINT,
  `reason` VARCHAR(40),
  PRIMARY KEY (`batch_id`, `job_id`, `attempt_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE,
  FOREIGN KEY (`instance_name`) REFERENCES instances(name)  
) ENGINE = InnoDB;
CREATE INDEX `attempts_instance_name` ON `attempts` (`instance_name`);

CREATE TABLE IF NOT EXISTS `ready_cores` (
  ready_cores_mcpu INT NOT NULL
) ENGINE = InnoDB;

INSERT INTO ready_cores (ready_cores_mcpu) VALUES (0);

CREATE TABLE IF NOT EXISTS `gevents_mark` (
  mark VARCHAR(40)
) ENGINE = InnoDB;

INSERT INTO `gevents_mark` (mark) VALUES (NULL);

CREATE TABLE IF NOT EXISTS `job_parents` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `parent_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `parent_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX job_parents_parent_id ON `job_parents` (batch_id, parent_id);

CREATE TABLE IF NOT EXISTS `job_attributes` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `key` VARCHAR(100) NOT NULL,
  `value` VARCHAR(65535),
  PRIMARY KEY (`batch_id`, `job_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX job_attributes_key_value ON `job_attributes` (`key`, `value`(256));

CREATE TABLE IF NOT EXISTS `batch_attributes` (
  `batch_id` BIGINT NOT NULL,
  `key` VARCHAR(100) NOT NULL,
  `value` VARCHAR(65535),
  PRIMARY KEY (`batch_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE  
) ENGINE = InnoDB;
CREATE INDEX batch_attributes_key_value ON `batch_attributes` (`key`, `value`(256));

DELIMITER $$

CREATE TRIGGER attempts_before_update BEFORE UPDATE ON attempts
FOR EACH ROW
BEGIN
  IF OLD.start_time IS NOT NULL AND (NEW.start_time IS NULL OR NEW.start_time < OLD.start_time) THEN
    SET NEW.start_time = OLD.start_time;
  END IF;

  IF OLD.end_time IS NOT NULL AND (NEW.end_time IS NULL OR NEW.end_time > OLD.end_time) THEN
    SET NEW.end_time = OLD.end_time;
    SET NEW.reason = OLD.reason;
  END IF;
END $$

CREATE TRIGGER attempts_after_update AFTER UPDATE ON attempts
FOR EACH ROW
BEGIN
  DECLARE job_cores_mcpu INT;
  DECLARE msec_diff BIGINT;
  DECLARE msec_mcpu_diff BIGINT;

  SELECT cores_mcpu INTO job_cores_mcpu FROM jobs
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;

  SET msec_diff = (GREATEST(COALESCE(NEW.end_time - NEW.start_time, 0), 0) -
                   GREATEST(COALESCE(OLD.end_time - OLD.start_time, 0), 0));

  SET msec_mcpu_diff = msec_diff * job_cores_mcpu;

  UPDATE batches
  SET msec_mcpu = batches.msec_mcpu + msec_mcpu_diff
  WHERE id = NEW.batch_id;

  UPDATE jobs
  SET msec_mcpu = jobs.msec_mcpu + msec_mcpu_diff
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id;
END $$

CREATE TRIGGER jobs_after_insert AFTER INSERT ON jobs
FOR EACH ROW
BEGIN
  DECLARE in_user VARCHAR(100);

  SELECT user INTO in_user from batches
  WHERE id = NEW.batch_id;

  IF NEW.state = 'Ready' THEN
    UPDATE user_resources
    SET n_ready_jobs = n_ready_jobs + 1, ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user;
  END IF;

  IF NEW.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs + 1, running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user;
  END IF;  
END $$

CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE in_user VARCHAR(100);

  SELECT user INTO in_user from batches
  WHERE id = NEW.batch_id;

  IF OLD.state = 'Ready' THEN
    UPDATE user_resources
    SET n_ready_jobs = n_ready_jobs - 1, ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu
    WHERE user = in_user;
  END IF;

  IF NEW.state = 'Ready' THEN
    UPDATE user_resources
    SET n_ready_jobs = n_ready_jobs + 1, ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user;
  END IF;

  IF OLD.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs - 1, running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu
    WHERE user = in_user;
  END IF;

  IF NEW.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs + 1, running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user;
  END IF;  
END $$

CREATE PROCEDURE activate_instance(
  IN in_instance_name VARCHAR(100),
  IN in_ip_address VARCHAR(100)
)
BEGIN
  DECLARE cur_state VARCHAR(40);
  DECLARE cur_token VARCHAR(100);

  START TRANSACTION;

  SELECT state, token INTO cur_state, cur_token FROM instances
  WHERE name = in_instance_name;

  IF cur_state = 'pending' THEN
    UPDATE instances
    SET state = 'active',
      activation_token = NULL,
      ip_address = in_ip_address WHERE name = in_instance_name;
    COMMIT;
    SELECT 0 as rc, cur_token as token;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not pending' as message;
  END IF;
END $$

CREATE PROCEDURE deactivate_instance(
  IN in_instance_name VARCHAR(100),
  IN in_reason VARCHAR(40),
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE name = in_instance_name;

  UPDATE attempts
  SET end_time = in_timestamp, reason = in_reason
  WHERE instance_name = in_instance_name;

  IF cur_state = 'pending' or cur_state = 'active' THEN
    UPDATE ready_cores
    SET ready_cores_mcpu = ready_cores_mcpu +
      COALESCE(
        (SELECT SUM(jobs.cores_mcpu)
         FROM attempts
         INNER JOIN jobs ON attempts.batch_id = jobs.batch_id AND attempts.job_id = jobs.job_id
         WHERE instance_name = in_instance_name),
        0);

    UPDATE jobs
    INNER JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id AND jobs.attempt_id = attempts.attempt_id
    SET state = 'Ready',
        jobs.attempt_id = NULL
    WHERE instance_name = in_instance_name;

    UPDATE attempts
    SET instance_name = NULL
    WHERE instance_name = in_instance_name;

    UPDATE instances SET state = 'inactive', free_cores_mcpu = cores_mcpu WHERE name = in_instance_name;

    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not live (active or pending)' as message;
  END IF;
END $$

CREATE PROCEDURE mark_instance_deleted(
  IN in_instance_name VARCHAR(100)
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE name = in_instance_name;

  IF cur_state = 'inactive' THEN
    UPDATE instances SET state = 'deleted' WHERE name = in_instance_name;
    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_state, 'state not inactive' as message;
  END IF;
END $$

CREATE PROCEDURE close_batch(
  IN in_batch_id BIGINT,
  IN in_timestamp BIGINT
)
BEGIN
  DECLARE cur_batch_closed BOOLEAN;
  DECLARE expected_n_jobs INT;
  DECLARE actual_n_jobs INT;

  START TRANSACTION;

  SELECT n_jobs, closed INTO expected_n_jobs, cur_batch_closed FROM batches
  WHERE id = in_batch_id AND NOT deleted;

  IF cur_batch_closed = 1 THEN
    COMMIT;
    SELECT 0 as rc;
  ELSEIF cur_batch_closed = 0 THEN
    SELECT COUNT(*) INTO actual_n_jobs FROM jobs
    WHERE batch_id = in_batch_id;

    IF actual_n_jobs = expected_n_jobs THEN
      UPDATE batches SET closed = 1 WHERE id = in_batch_id;
      UPDATE batches SET time_completed = in_timestamp
        WHERE id = in_batch_id AND n_completed = batches.n_jobs;
      UPDATE ready_cores
        SET ready_cores_mcpu = ready_cores_mcpu +
          COALESCE(
            (SELECT SUM(cores_mcpu) FROM jobs
             WHERE jobs.state = 'Ready' AND jobs.batch_id = in_batch_id),
            0);
      COMMIT;
      SELECT 0 as rc;
    ELSE
      ROLLBACK;
      SELECT 2 as rc, expected_n_jobs, actual_n_jobs, 'wrong number of jobs' as message;
    END IF;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_batch_closed, 'batch closed is not 0 or 1' as message;
  END IF;
END $$

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

  START TRANSACTION;

  SELECT state, cores_mcpu,
    (jobs.cancelled OR batches.cancelled) AND NOT always_run
  INTO cur_job_state, cur_cores_mcpu, cur_job_cancel
  FROM jobs
  INNER JOIN batches ON batches.id = jobs.batch_id
  WHERE batch_id = in_batch_id AND batches.closed
    AND job_id = in_job_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
    INSERT INTO attempts (batch_id, job_id, attempt_id, instance_name) VALUES (in_batch_id, in_job_id, in_attempt_id, in_instance_name);
    UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu - cur_cores_mcpu;
    UPDATE instances SET free_cores_mcpu = free_cores_mcpu - cur_cores_mcpu WHERE name = in_instance_name;
    COMMIT;
    SELECT 0 as rc, in_instance_name;
  ELSE
    ROLLBACK;
    SELECT 1 as rc,
      cur_job_state,
      cur_job_cancel,
      cur_instance_state,
      in_instance_name,
      'job not Ready or cancelled or instance not active' as message;
  END IF;
END $$

CREATE PROCEDURE unschedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN expected_instance_name VARCHAR(100),
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_instance_name VARCHAR(100);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_attempt_id VARCHAR(40);

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs WHERE batch_id = in_batch_id AND job_id = in_job_id;

  SELECT instance_name
  INTO cur_job_instance_name
  FROM attempts WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = cur_attempt_id;

  IF cur_job_state = 'Running' AND cur_job_instance_name = expected_instance_name THEN
    UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu + cur_cores_mcpu;
    UPDATE instances SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu WHERE name = cur_job_instance_name;
    UPDATE attempts
      SET end_time = new_end_time, reason = new_reason, instance_name = NULL
      WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = cur_attempt_id;
    UPDATE jobs SET state = 'Ready', attempt_id = NULL WHERE batch_id = in_batch_id AND job_id = in_job_id;      
    COMMIT;
    SELECT 0 as rc;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_job_state, cur_job_instance_name, expected_instance_name,
      'job state not Running or wrong instance' as message;
  END IF;
END $$

CREATE PROCEDURE mark_job_complete(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN new_state VARCHAR(40),
  IN new_status VARCHAR(65535),
  IN new_start_time BIGINT,
  IN new_end_time BIGINT,
  IN new_reason VARCHAR(40),
  IN new_timestamp BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_instance_name VARCHAR(100);
  DECLARE cur_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id;

  SELECT instance_name
  INTO cur_job_instance_name
  FROM attempts
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  IF cur_job_state = 'Ready' OR cur_job_state = 'Running' THEN    
    UPDATE jobs
    SET state = new_state, status = new_status, attempt_id = NULL
    WHERE batch_id = in_batch_id AND job_id = in_job_id;

    UPDATE batches SET n_completed = n_completed + 1 WHERE id = in_batch_id;
    UPDATE batches SET time_completed = new_timestamp
      WHERE id = in_batch_id AND n_completed = batches.n_jobs;

    IF new_state = 'Cancelled' THEN
      UPDATE batches SET n_cancelled = n_cancelled + 1 WHERE id = in_batch_id;
    ELSEIF new_state = 'Error' OR new_state = 'Failed' THEN
      UPDATE batches SET n_failed = n_failed + 1 WHERE id = in_batch_id;
    ELSE
      UPDATE batches SET n_succeeded = n_succeeded + 1 WHERE id = in_batch_id;
    END IF;

    IF cur_job_instance_name IS NOT NULL THEN
      UPDATE instances
      SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu
      WHERE name = cur_job_instance_name;
    END IF;

    IF cur_job_state = 'Ready' THEN
      UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu - cur_cores_mcpu;
    END IF;
    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu +
        COALESCE(
          (SELECT SUM(jobs.cores_mcpu) FROM jobs
           INNER JOIN `job_parents`
             ON jobs.batch_id = `job_parents`.batch_id AND
                jobs.job_id = `job_parents`.job_id
           WHERE jobs.batch_id = in_batch_id AND
                 `job_parents`.batch_id = in_batch_id AND
                 `job_parents`.parent_id = in_job_id AND
                 jobs.n_pending_parents = 1),
          0);

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

    IF in_attempt_id IS NOT NULL THEN
      UPDATE attempts
      SET start_time = new_start_time, end_time = new_end_time, reason = new_reason, instance_name = NULL
      WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;
    END IF;

    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state,
      cur_cores_mcpu as cores_mcpu,
      cur_job_instance_name as instance_name;
  ELSEIF cur_job_state = 'Cancelled' OR cur_job_state = 'Error' OR
         cur_job_state = 'Failed' OR cur_job_state = 'Success' THEN
    COMMIT;
    SELECT 0 as rc,
      cur_job_state as old_state;
  ELSE
    ROLLBACK;
    SELECT 1 as rc, cur_job_state, 'job state not Ready, Running or complete' as message;
  END IF;
END $$

DELIMITER ;
