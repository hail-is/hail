CREATE TABLE IF NOT EXISTS `instance_healthchecks` (
  `name` VARCHAR(100) NOT NULL,
  `failed_request_count` INT NOT NULL DEFAULT 0,
  `last_updated` BIGINT NOT NULL,
  PRIMARY KEY (`name`),
  FOREIGN KEY (`name`) REFERENCES instances(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO instance_healthchecks (name, failed_request_count, last_updated)
  SELECT name, failed_request_count, last_updated FROM instances;

ALTER TABLE instances DROP COLUMN failed_request_count, DROP COLUMN last_updated;
