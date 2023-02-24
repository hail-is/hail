DELIMITER $$

DROP TRIGGER IF EXISTS attempt_resources_before_insert $$
CREATE TRIGGER attempt_resources_before_insert BEFORE INSERT ON attempt_resources
FOR EACH ROW
BEGIN
  DECLARE cur_resource VARCHAR(100);
  SELECT resource INTO cur_resource FROM resources WHERE resource_id = NEW.resource_id;
  SET NEW.resource = cur_resource;
END $$

DELIMITER ;
