ALTER TABLE user_resources ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE user_resources DROP PRIMARY KEY, ADD PRIMARY KEY(`user`, `token`);

ALTER TABLE ready_cores ADD COLUMN token INT NOT NULL DEFAULT 0;
ALTER TABLE ready_cores ADD PRIMARY KEY(`token`);

DELIMITER $$

DROP PROCEDURE IF EXISTS insert_ready_cores_tokens;
CREATE PROCEDURE insert_ready_cores_tokens()
BEGIN
    DECLARE i int DEFAULT 0;
    WHILE i < 32 DO
        INSERT IGNORE INTO ready_cores (token) VALUES (i);
        SET i = i + 1;
    END WHILE;
END $$

DROP PROCEDURE IF EXISTS insert_user_resources_tokens;
CREATE PROCEDURE insert_user_resources_tokens(
  IN in_user VARCHAR(100)
)
BEGIN
    DECLARE i int DEFAULT 0;
    WHILE i < 32 DO
        INSERT IGNORE INTO user_resources (user, token) VALUES (in_user, i);
        SET i = i + 1;
    END WHILE;
END $$

DROP TRIGGER IF EXISTS jobs_after_update;
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE in_user VARCHAR(100);
  DECLARE rand_token INT;

  SELECT user INTO in_user from batches
  WHERE id = NEW.batch_id;

  SET rand_token = FLOOR(RAND() * 32);

  IF OLD.state = 'Ready' THEN
    UPDATE user_resources
      SET n_ready_jobs = n_ready_jobs - 1, ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu
      WHERE user = in_user AND token = rand_token;

    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu - OLD.cores_mcpu
      WHERE token = rand_token;
  END IF;

  IF NEW.state = 'Ready' THEN
    UPDATE user_resources
      SET n_ready_jobs = n_ready_jobs + 1, ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
      WHERE user = in_user AND token = rand_token;

    UPDATE ready_cores
      SET ready_cores_mcpu = ready_cores_mcpu + NEW.cores_mcpu
      WHERE token = rand_token;
  END IF;

  IF OLD.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs - 1, running_cores_mcpu = running_cores_mcpu - OLD.cores_mcpu
    WHERE user = in_user AND token = rand_token;
  END IF;

  IF NEW.state = 'Running' THEN
    UPDATE user_resources
    SET n_running_jobs = n_running_jobs + 1, running_cores_mcpu = running_cores_mcpu + NEW.cores_mcpu
    WHERE user = in_user AND token = rand_token;
  END IF;
END $$
