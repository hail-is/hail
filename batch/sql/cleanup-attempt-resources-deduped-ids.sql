DROP TRIGGER IF EXISTS attempt_resources_before_insert;

ALTER TABLE attempt_resources DROP COLUMN deduped_resource_id, ALGORITHM=INPLACE, LOCK=NONE;
