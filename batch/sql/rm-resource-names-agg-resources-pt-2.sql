DROP TRIGGER IF EXISTS attempt_resources_before_insert;
ALTER TABLE attempt_resources DROP COLUMN resource, ALGORITHM=INPLACE, LOCK=NONE;
