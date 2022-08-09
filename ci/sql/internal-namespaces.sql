CREATE TABLE IF NOT EXISTS `internal_namespaces` (
  `namespace_name` VARCHAR(100) NOT NULL,
  `creation_time` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `expiration_time` TIMESTAMP,
  `services` VARCHAR(10000) NOT NULL,
  PRIMARY KEY (`namespace_name`)
) ENGINE = InnoDB;
