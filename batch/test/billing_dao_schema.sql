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

CREATE TABLE IF NOT EXISTS `quote_events` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `quote_id` INT NOT NULL,
  `timestamp` BIGINT NOT NULL,
  `actor` VARCHAR(100) NOT NULL,
  `action` VARCHAR(100) NOT NULL,
  `target_user` VARCHAR(100) DEFAULT NULL,
  `target_project` VARCHAR(100) DEFAULT NULL,
  `detail` TEXT DEFAULT NULL,
  `comment` VARCHAR(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `quote_managers` (
  `quote_id` INT NOT NULL,
  `user`     VARCHAR(100) NOT NULL,
  `role`     ENUM('owner', 'manager') NOT NULL DEFAULT 'manager',
  PRIMARY KEY (`quote_id`, `user`),
  FOREIGN KEY (`quote_id`) REFERENCES `quotes`(`id`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `billing_projects` (
  `name`             VARCHAR(100) NOT NULL,
  `name_cs`          VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  `status`           ENUM('open', 'closed', 'deleted') NOT NULL DEFAULT 'open',
  `limit`            DOUBLE DEFAULT NULL,
  `msec_mcpu`        BIGINT DEFAULT 0,
  `quote_id`         INT DEFAULT 1,
  `low_budget_alert` DOUBLE DEFAULT NULL,
  PRIMARY KEY (`name`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `billing_project_users` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user`            VARCHAR(100) NOT NULL,
  `user_cs`         VARCHAR(100) NOT NULL COLLATE utf8mb4_0900_as_cs,
  PRIMARY KEY (`billing_project`, `user`),
  FOREIGN KEY (`billing_project`) REFERENCES `billing_projects`(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `billing_project_events` (
  `id`              BIGINT NOT NULL AUTO_INCREMENT,
  `billing_project` VARCHAR(100) NOT NULL,
  `timestamp`       BIGINT NOT NULL,
  `actor`           VARCHAR(100) NOT NULL,
  `action`          VARCHAR(100) NOT NULL,
  `target_user`     VARCHAR(100) DEFAULT NULL,
  `detail`          TEXT DEFAULT NULL,
  `comment`         VARCHAR(255) DEFAULT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`billing_project`) REFERENCES `billing_projects`(`name`) ON DELETE CASCADE
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `resources` (
  `resource`            VARCHAR(100) NOT NULL,
  `rate`                DOUBLE NOT NULL,
  `resource_id`         INT AUTO_INCREMENT UNIQUE NOT NULL,
  `deduped_resource_id` INT DEFAULT NULL,
  PRIMARY KEY (`resource`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `batches` (
  `id`              BIGINT NOT NULL AUTO_INCREMENT,
  `billing_project` VARCHAR(100) NOT NULL,
  `time_completed`  BIGINT DEFAULT NULL,
  `deleted`         BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `aggregated_billing_project_user_resources_v3` (
  `billing_project` VARCHAR(100) NOT NULL,
  `user`            VARCHAR(100) NOT NULL,
  `resource_id`     INT NOT NULL,
  `token`           INT NOT NULL,
  `usage`           BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (`billing_project`, `user`, `resource_id`, `token`)
) ENGINE = InnoDB;

INSERT INTO `quotes` (`name`, `name_cs`, `cost_object`, `time_created`)
VALUES ('INTERNAL', 'INTERNAL', 'INTERNAL', 0);
