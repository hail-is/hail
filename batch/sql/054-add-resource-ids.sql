ALTER TABLE `resources` ADD COLUMN resource_id INT AUTO_INCREMENT UNIQUE NOT NULL;
ALTER TABLE `attempt_resources` ADD COLUMN resource_id INT, ALGORITHM=INSTANT;

SET foreign_key_checks = 0;
ALTER TABLE `attempt_resources` ADD FOREIGN KEY (resource_id) REFERENCES `resources` (resource_id) ON DELETE CASCADE, ALGORITHM=INPLACE;
SET foreign_key_checks = 1;

DELIMITER $$

DROP TRIGGER IF EXISTS attempt_resources_before_insert $$
CREATE TRIGGER attempt_resources_before_insert BEFORE INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_resource_id INT;
  SELECT resource_id INTO cur_resource_id FROM resources WHERE resource = NEW.resource;
  SET NEW.resource_id = cur_resource_id;
END $$

DELIMITER ;
