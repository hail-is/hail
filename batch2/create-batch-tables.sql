CREATE TABLE IF NOT EXISTS `batch` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `userdata` VARCHAR(65535) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `attributes` VARCHAR(65535),
  `callback` VARCHAR(65535),
  `deleted` BOOLEAN NOT NULL default false,
  `cancelled` BOOLEAN NOT NULL default false,
  `closed` BOOLEAN NOT NULL default false,
  `n_jobs` INT NOT NULL default 0,
  `n_completed` INT NOT NULL default 0,
  `n_succeeded` INT NOT NULL default 0,
  `n_failed` INT NOT NULL default 0,
  `n_cancelled` INT NOT NULL default 0,
  `time_created` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
CREATE INDEX `batch_user` ON `batch` (`user`);
CREATE INDEX `batch_deleted` ON `batch` (`deleted`);

CREATE TABLE IF NOT EXISTS `jobs` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `state` VARCHAR(40) NOT NULL,
  `directory` VARCHAR(1024),
  `spec` VARCHAR(65535) NOT NULL,
  `cores_mcpu` INT NOT NULL,
  `instance_id` BIGINT,
  `status` VARCHAR(65535),
  `n_pending_parents` INT NOT NULL,
  `cancel` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batch(id) ON DELETE CASCADE,
  FOREIGN KEY (`instance_id`) REFERENCES instances(id) ON DELETE SET NULL
) ENGINE = InnoDB;
CREATE INDEX `jobs_state` ON `jobs` (`state`);
CREATE INDEX `jobs_instance_id` ON `jobs` (`instance_id`);

CREATE TABLE IF NOT EXISTS `ready_cores` (
  ready_cores_mcpu INT NOT NULL
) ENGINE = InnoDB;

INSERT INTO ready_cores (ready_cores_mcpu) VALUES (0);

CREATE TABLE IF NOT EXISTS `jobs-parents` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `parent_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `parent_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batch(id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX jobs_parents_parent_id ON `jobs-parents` (batch_id, parent_id);

CREATE TABLE IF NOT EXISTS `batch-attributes` (
  `batch_id` BIGINT NOT NULL,
  `key` VARCHAR(100) NOT NULL,
  `value` VARCHAR(65535),
  PRIMARY KEY (`batch_id`, `key`),
  FOREIGN KEY (`batch_id`) REFERENCES batch(id) ON DELETE CASCADE  
) ENGINE = InnoDB;
CREATE INDEX batch_attributes_key_value ON `batch-attributes` (`key`, `value`(256));

CREATE TABLE IF NOT EXISTS `instances` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `state` VARCHAR(40) NOT NULL,
  `name` VARCHAR(100) NOT NULL,
  `token` VARCHAR(100) NOT NULL,
  `capacity_mcpu` INT NOT NULL,
  `free_cores_mcpu` INT NOT NULL,
  `ip_address` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;

DELIMITER $$

CREATE PROCEDURE activate_instance(
  IN in_instance_id BIGINT,
  IN in_ip_address VARCHAR(100),
  OUT success INT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE id = in_instance_id;

  IF cur_state = 'pending' THEN
    UPDATE instances SET state = 'active', ip_address = in_ip_address WHERE id = in_instance_id;
    COMMIT;
    SET success = 1;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

CREATE PROCEDURE deactivate_instance(
  IN in_instance_id BIGINT,
  OUT success INT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE id = in_instance_id;

  IF cur_state = 'pending' or cur_state = 'active' THEN
    UPDATE instances SET state = 'inactive', free_cores_mcpu = cores_mcpu WHERE id = in_instance_id;
    UPDATE jobs SET instance_id = NULL where instance_id = in_instance_id;
    COMMIT;
    SET success = 1;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

CREATE PROCEDURE mark_instance_deleted(
  IN in_instance_id BIGINT,
  OUT success INT
)
BEGIN
  DECLARE cur_state VARCHAR(40);

  START TRANSACTION;

  SELECT state INTO cur_state FROM instances WHERE id = in_instance_id;

  IF cur_state = 'inactive' THEN
    UPDATE instances SET state = 'deleted' WHERE id = in_instance_id;
    COMMIT;
    SET success = 1;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

CREATE PROCEDURE schedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id BIGINT,
  IN in_instance_id BIGINT,
  OUT success INT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);

  START TRANSACTION;

  SELECT state, cores_mcpu,
  INTO cur_job_state, cur_cores_mcpu,
  FROM jobs WHERE batch_id = in_batch_id AND job_id = in_job_id;

  SELECT state INTO cur_instance_state FROM instances WHERE id = in_instance_id;

  IF cur_job_state = 'Ready' AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', instance_id = in_instance_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
    UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu - cur_cores_mcpu;
    UPDATE instances SET free_cores_mcpu = free_cores_mcpu - cur_cores_mcpu WHERE id = in_instance_id;
    COMMIT;
    SET success = 1;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

CREATE PROCEDURE unschedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id BIGINT,
  OUT success INT,
  OUT out_instance_id BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_instance_id BIGINT;
  DECLARE cur_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu, instance_id
  INTO cur_job_state, cur_cores_mcpu, cur_job_instance_id
  FROM jobs WHERE batch_id = in_batch_id AND job_id = in_job_id;

  IF cur_job_state = 'Running' THEN
    UPDATE jobs SET state = 'Ready', instance_id = NULL WHERE batch_id = in_batch_id AND job_id = in_job_id;
    UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu + cur_cores_mcpu;
    UPDATE instances SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu WHERE id = cur_job_instance_id;
    COMMIT;
    SET success = 1;
    SET out_instance_id = cur_job_instance_id;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

CREATE PROCEDURE mark_job_complete(
  IN in_batch_id BIGINT,
  IN in_job_id BIGINT,
  IN new_state VARCHAR(40),
  IN new_status VARCHAR(65535),
  OUT success INT,
  OUT out_instance_id BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_instance_id BIGINT;
  DECLARE cur_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu, instance_id
  INTO cur_job_state, cur_cores_mcpu, cur_job_instance_id
  FROM jobs WHERE batch_id = in_batch_id AND job_id = in_job_id;

  IF cur_job_state = 'Ready' OR cur_job_state = 'Running' THEN
    UPDATE jobs SET state = new_state, status = new_state, instance_id = NULL;
    IF cur_job_state = 'Ready' THEN
      UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu - cur_cores_mcpu;
    END IF;
    UPDATE batch SET n_completed = n_completed + 1 WHERE id = in_batch_id;
    IF new_state = 'Cancelled' THEN
      UPDATE batch SET n_cancelled = n_cancelled + 1 WHERE id = in_batch_id;
    ELSEIF new_state = 'Error' OR new_state = 'Failed':
      UPDATE batch SET n_failed = n_failed + 1 WHERE id = in_batch_id;
    ELSE
      UPDATE batch SET n_succeeded = n_suceeded + 1 WHERE id = in_batch_id;
    END IF;
    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu + (
        SELECT SUM(cores_mcpu) FROM jobs
	WHERE `jobs-parents`.batch_id = in_batch_id AND
	      `job-parents`.parent_id = in_job_id);
    UPDATE jobs SET n_pending_parents = n_pending_parents - 1,
                    state = IF(n_pending_parents = 1, 'Ready', 'Pending'),
                    cancel = IF(new_state = 'Success', cancel, 1)
      WHERE jobs.batch_id = in_batch_id AND
        EXISTS (SELECT * FROM `jobs-parents`
                WHERE `jobs-parents`.batch_id = in_batch_id AND
                  `jobs-parents`.job_id = jobs.job_id AND
                  `job-parents`.parent_id = in_job_id);
    UPDATE ready_cores SET ready_cores_mcpu = ready_cores_mcpu - cur_cores_mcpu;
    
    IF cur_job_instance_id IS NOT NULL THEN
      UPDATE instances SET free_cores_mcpu = free_cores_mcpu + cur_cores_mcpu WHERE id = cur_job_instance_id;
    END IF;
    COMMIT;
    SET success = 1;
    SET out_instance_id = cur_job_instance_id;
  ELSE
    ROLLBACK;
    SET success = 0;
  END IF;
END $$

DELIMITER ;
