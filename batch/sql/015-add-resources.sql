CREATE TABLE IF NOT EXISTS `resources` (
  `resource` VARCHAR(100) NOT NULL,
  `rate` DOUBLE NOT NULL,
  PRIMARY KEY (`resource`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_batch_resources` (
  `batch_id` BIGINT NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `resource`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_job_resources` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `job_id`, `resource`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `attempt_resources` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `attempt_id` VARCHAR(40) NOT NULL,
  `resource` VARCHAR(100) NOT NULL,
  `quantity` BIGINT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `attempt_id`, `resource`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`, `attempt_id`) REFERENCES attempts(`batch_id`, `job_id`, `attempt_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource`) REFERENCES resources(`resource`) ON DELETE CASCADE
) ENGINE = InnoDB;

DROP INDEX batches_time_created ON `batches`;
CREATE INDEX `batches_time_completed` ON `batches` (`time_completed`);

DELIMITER $$

DROP TRIGGER IF EXISTS attempt_resources_after_insert $$
CREATE TRIGGER attempt_resources_after_insert AFTER INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_start_time BIGINT;
  DECLARE cur_end_time BIGINT;
  DECLARE msec_diff BIGINT;

  SELECT start_time, end_time INTO cur_start_time, cur_end_time
  FROM attempts
  WHERE batch_id = NEW.batch_id AND job_id = NEW.job_id AND attempt_id = NEW.attempt_id
  LOCK IN SHARE MODE;

  SET msec_diff = GREATEST(COALESCE(cur_end_time - cur_start_time, 0), 0);

  INSERT INTO aggregated_job_resources (batch_id, job_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.job_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;

  INSERT INTO aggregated_batch_resources (batch_id, resource, `usage`)
  VALUES (NEW.batch_id, NEW.resource, NEW.quantity * msec_diff)
  ON DUPLICATE KEY UPDATE
    `usage` = `usage` + NEW.quantity * msec_diff;
END $$


DROP TRIGGER IF EXISTS attempts_after_update $$
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

  UPDATE aggregated_batch_resources
  JOIN (SELECT batch_id, resource, quantity
        FROM attempt_resources
        WHERE batch_id = NEW.batch_id AND
        job_id = NEW.job_id AND
        attempt_id = NEW.attempt_id) AS t
  ON aggregated_batch_resources.batch_id = t.batch_id AND
    aggregated_batch_resources.resource = t.resource
  SET `usage` = `usage` + msec_diff * t.quantity
  WHERE aggregated_batch_resources.batch_id = NEW.batch_id;

  UPDATE aggregated_job_resources
  JOIN (SELECT batch_id, job_id, resource, quantity
        FROM attempt_resources
        WHERE batch_id = NEW.batch_id AND
        job_id = NEW.job_id AND
        attempt_id = NEW.attempt_id) AS t
  ON aggregated_job_resources.batch_id = t.batch_id AND
    aggregated_job_resources.job_id = t.job_id AND
    aggregated_job_resources.resource = t.resource
  SET `usage` = `usage` + msec_diff * t.quantity
  WHERE aggregated_job_resources.batch_id = NEW.batch_id AND aggregated_job_resources.job_id = NEW.job_id;
END $$

DELIMITER ;
