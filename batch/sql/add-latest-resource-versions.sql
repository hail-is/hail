CREATE TABLE IF NOT EXISTS `latest_resource_versions` (
  `prefix` VARCHAR(100) NOT NULL,
  `version` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`prefix`)
) ENGINE = InnoDB;
