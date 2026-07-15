CREATE TABLE IF NOT EXISTS `quotes` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `name_cs` VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  `cost_object` VARCHAR(255) NOT NULL,
  `authorized_amount` DOUBLE DEFAULT NULL,
  `pi_name` VARCHAR(255) DEFAULT NULL,
  `pm_designee` VARCHAR(255) DEFAULT NULL,
  `time_created` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
CREATE UNIQUE INDEX `quote_name` ON `quotes` (`name`);
CREATE UNIQUE INDEX `quote_name_cs` ON `quotes` (`name_cs`);

CREATE TABLE IF NOT EXISTS `quote_managers` (
  `quote_id` INT NOT NULL,
  `user` VARCHAR(100) NOT NULL,
  `role` ENUM('owner', 'manager') NOT NULL DEFAULT 'manager',
  PRIMARY KEY (`quote_id`, `user`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `quote_managers_user` ON `quote_managers` (`user`);

CREATE TABLE IF NOT EXISTS `quote_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `quote_id` INT NOT NULL,
  `timestamp` BIGINT NOT NULL,
  `actor` VARCHAR(100) NOT NULL,
  `action` VARCHAR(100) NOT NULL,
  `target_user` VARCHAR(100) DEFAULT NULL,
  `target_project` VARCHAR(100) DEFAULT NULL,
  `detail` TEXT DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `quote_events_quote_id_timestamp` ON `quote_events` (`quote_id`, `timestamp`);

CREATE TABLE IF NOT EXISTS `billing_project_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `billing_project` VARCHAR(100) NOT NULL,
  `timestamp` BIGINT NOT NULL,
  `actor` VARCHAR(100) NOT NULL,
  `action` VARCHAR(100) NOT NULL,
  `target_user` VARCHAR(100) DEFAULT NULL,
  `detail` TEXT DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`billing_project`) REFERENCES `billing_projects`(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `billing_project_events_bp_timestamp` ON `billing_project_events` (`billing_project`, `timestamp`);

INSERT INTO `quotes` (`name`, `name_cs`, `cost_object`, `time_created`)
VALUES ('INTERNAL', 'INTERNAL', 'INTERNAL', UNIX_TIMESTAMP() * 1000);

ALTER TABLE `billing_projects`
  ADD COLUMN `quote_id` INT DEFAULT 1,
  ADD COLUMN `low_budget_alert` DOUBLE DEFAULT NULL,
  ADD CONSTRAINT `fk_billing_projects_quote_id` FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`);
