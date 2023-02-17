ALTER TABLE resources ADD COLUMN deduped_resource_id INT DEFAULT NULL, ALGORITHM=INSTANT;

--  don't need the migrated flag as the billing trigger is only on inserting into attempt_resources
ALTER TABLE attempt_resources ADD COLUMN deduped_resource_id INT DEFAULT NULL, ALGORITHM=INSTANT;

ALTER TABLE aggregated_billing_project_user_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_billing_project_user_resources_by_date_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_batch_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;
ALTER TABLE aggregated_job_resources_v2 ADD COLUMN migrated BOOLEAN DEFAULT FALSE, ALGORITHM=INSTANT;

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_v3` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_v3 ON `aggregated_billing_project_user_resources_v3` (`user`);

DROP TABLE IF EXISTS `aggregated_billing_project_user_resources_by_date_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_by_date_v3` (
  `billing_date` DATE NOT NULL,
  `billing_project` VARCHAR(100) NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_date`, `billing_project`, `user`, `resource_id`, `token`),
  FOREIGN KEY (`billing_project`) REFERENCES billing_projects(name) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX aggregated_billing_project_user_resources_by_date_v3_user ON `aggregated_billing_project_user_resources_by_date_v3` (`billing_date`, `user`);

DROP TABLE IF EXISTS `aggregated_batch_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_batch_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `resource_id` INT NOT NULL,
  `token` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `resource_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DROP TABLE IF EXISTS `aggregated_job_resources_v3`;
CREATE TABLE IF NOT EXISTS `aggregated_job_resources_v3` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `resource_id` INT NOT NULL,
  `usage` BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`batch_id`, `job_id`, `resource_id`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(`batch_id`, `job_id`) ON DELETE CASCADE,
  FOREIGN KEY (`resource_id`) REFERENCES resources(`resource_id`) ON DELETE CASCADE
) ENGINE = InnoDB;

DELIMITER $$

DROP TRIGGER IF EXISTS resources_before_insert $$
CREATE TRIGGER resources_before_insert BEFORE INSERT ON resources
FOR EACH ROW
BEGIN
  SET NEW.deduped_resource_id = NEW.resource_id;
END $$

-- This is okay here because the deduplication mitigation in batch will have already merged
CREATE TRIGGER attempt_resources_before_insert BEFORE INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  SET NEW.deduped_resource_id = NEW.resource_id;
END $$

DELIMITER ;
