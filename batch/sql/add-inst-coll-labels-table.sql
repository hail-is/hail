CREATE TABLE IF NOT EXISTS `inst_coll_labels` (
  `name` VARCHAR(255) NOT NULL,
  `label` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`name`, `label`),
  FOREIGN KEY (`name`) REFERENCES inst_colls(name) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO inst_coll_labels (`name`, `label`)
SELECT `name`, `label` FROM pools
WHERE `label` != "";
