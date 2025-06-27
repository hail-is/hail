CREATE TABLE IF NOT EXISTS `active_namespaces` (
  `namespace` VARCHAR(100) NOT NULL,
  `creation_time` TIMESTAMP NOT NULL DEFAULT (UTC_TIMESTAMP),
  `expiration_time` TIMESTAMP,
  PRIMARY KEY (`namespace`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `deployed_services` (
  `namespace` VARCHAR(100) NOT NULL,
  `service` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`namespace`, `service`),
  FOREIGN KEY (`namespace`) REFERENCES active_namespaces(namespace) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO `active_namespaces` (`namespace`) VALUES ('default');
INSERT INTO `deployed_services` (`namespace`, `service`) VALUES
('default', 'auth'), ('default', 'batch'), ('default', 'batch-driver'), ('default', 'ci');
