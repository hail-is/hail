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
