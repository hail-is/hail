ALTER TABLE job_groups MODIFY COLUMN `state` ENUM('running', 'complete', 'open'), ALGORITHM=INSTANT;
