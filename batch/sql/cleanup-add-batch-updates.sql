START TRANSACTION;

ALTER TABLE jobs MODIFY COLUMN `update_id` INT NOT NULL;
ALTER TABLE batches_inst_coll_staging MODIFY COLUMN `update_id` INT NOT NULL;
ALTER TABLE batch_inst_coll_cancellable_resources MODIFY COLUMN `update_id` INT NOT NULL;

SET foreign_key_checks = 0;
ALTER TABLE jobs ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batches_inst_coll_staging ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
ALTER TABLE batch_inst_coll_cancellable_resources ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE, ALGORITHM=INPLACE;
SET foreign_key_checks = 1;

ALTER TABLE batches_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;
ALTER TABLE batch_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;

DROP TRIGGER IF EXISTS batches_before_insert;
DROP TRIGGER IF EXISTS batches_after_insert;
DROP TRIGGER IF EXISTS batches_before_update;
DROP TRIGGER IF EXISTS batches_after_update;

ALTER TABLE batches DROP COLUMN update_added, ALGORITHM=INPLACE, LOCK=NONE;

COMMIT;
