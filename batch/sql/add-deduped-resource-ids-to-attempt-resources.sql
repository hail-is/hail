ALTER TABLE attempt_resources ADD COLUMN deduped_resource_id INT DEFAULT NULL, ALGORITHM=INSTANT;
