CREATE INDEX job_attributes_batch_id_key_value ON `job_attributes` (batch_id, `key`, `value`(256));
CREATE INDEX job_attributes_value ON `job_attributes` (batch_id, `value`(256));
