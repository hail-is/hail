DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS rename_job_groups_cancelled_column(
  IN from_column VARCHAR(64), 
  IN to_column VARCHAR(64)
)
rename_column:BEGIN
  DECLARE v_constraint_name VARCHAR(64);
  DECLARE v_done INT DEFAULT FALSE;
  DECLARE v_error_message TEXT;
  DECLARE v_primary_exists INT DEFAULT FALSE;
  DECLARE v_sqlstate CHAR(5);

  -- Dynamically fetch the names of contraints related to `from_column` 
  DECLARE v_cursor CURSOR FOR
    SELECT CONSTRAINT_NAME
      FROM information_schema.KEY_COLUMN_USAGE
     WHERE TABLE_NAME = 'job_groups_cancelled'
       AND COLUMN_NAME = from_column;
  
  -- Rollback handler if something goes wrong
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    GET DIAGNOSTICS CONDITION 1
        v_sqlstate = RETURNED_SQLSTATE,
        v_error_message = MESSAGE_TEXT;

    SELECT CONCAT('Error SQLSTATE: ', v_sqlstate, ', Message: ', v_error_message) AS ErrorDetails;

    ROLLBACK;
  END;

  DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = TRUE;

  -- Check if `from_column` exists and `to_column` does not exist
  IF NOT EXISTS (SELECT * FROM information_schema.COLUMNS 
                  WHERE TABLE_NAME = 'job_groups_cancelled' 
                    AND COLUMN_NAME = from_column) THEN
    SELECT CONCAT('Error: Column ', from_column, ' does not exist.') AS ErrorDetails;
    LEAVE rename_column;
  END IF;

  IF EXISTS (SELECT * FROM information_schema.COLUMNS 
              WHERE TABLE_NAME = 'job_groups_cancelled' 
                AND COLUMN_NAME = to_column) THEN
    SELECT CONCAT('Error: Column ', to_column, ' already exists.') AS ErrorDetails;
    LEAVE rename_column;
  END IF;

  START TRANSACTION;

  OPEN v_cursor;
  
  drop_fk_loop: LOOP
  
    FETCH v_cursor INTO v_constraint_name;

    IF v_done THEN
      LEAVE drop_fk_loop;
    END IF;

    SELECT v_constraint_name;
	
    IF v_constraint_name = 'PRIMARY' THEN
      SET v_primary_exists = TRUE;
    ELSE
      SET @sql = CONCAT('ALTER TABLE job_groups_cancelled DROP FOREIGN KEY ', v_constraint_name);
      PREPARE stmt FROM @sql;
      EXECUTE stmt;
      DEALLOCATE PREPARE stmt;
    END IF;

  END LOOP;

  CLOSE v_cursor;

  -- drop primary key
  IF v_primary_exists THEN
    SET @sql = 'ALTER TABLE job_groups_cancelled DROP PRIMARY KEY';
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
  END IF;

  -- rename column from `from_column` to `to_column`
  SET @sql = CONCAT('ALTER TABLE job_groups_cancelled CHANGE COLUMN `', from_column, '` `', to_column, '` BIGINT NOT NULL');
  PREPARE stmt FROM @sql;
  EXECUTE stmt;
  DEALLOCATE PREPARE stmt;

  -- Recreate the primary and foreign key constraints using `to_column`
  SET @sql = CONCAT('ALTER TABLE job_groups_cancelled ADD PRIMARY KEY (`', to_column, '`, `job_group_id`), ',
                    'ADD FOREIGN KEY (`', to_column, '`) REFERENCES batches(`id`) ON DELETE CASCADE, ',
                    'ADD FOREIGN KEY (`', to_column, '`, `job_group_id`) REFERENCES job_groups(`batch_id`, `job_group_id`) ON DELETE CASCADE');
  PREPARE stmt FROM @sql;
  EXECUTE stmt;
  DEALLOCATE PREPARE stmt;

  COMMIT;

END $$

DELIMITER ;
