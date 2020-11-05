CREATE TABLE IF NOT EXISTS `atgu_resources` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `time_created` BIGINT NOT NULL,
  `title` TEXT NOT NULL,
  `description` TEXT NOT NULL,
  `contents` LONGTEXT NOT NULL,
  `tags` TEXT NOT NULL,
  `attachments` TEXT NOT NULL,
  `time_updated` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE = InnoDB;
