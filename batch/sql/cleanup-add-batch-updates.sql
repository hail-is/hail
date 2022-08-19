START TRANSACTION;

ALTER TABLE batches_inst_coll_staging MODIFY COLUMN `update_id` VARCHAR(40) NOT NULL;
ALTER TABLE batch_inst_coll_cancellable_resources MODIFY COLUMN `update_id` VARCHAR(40) NOT NULL;

ALTER TABLE batches_inst_coll_staging ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE;
ALTER TABLE batch_inst_coll_cancellable_resources ADD FOREIGN KEY (`batch_id`, `update_id`) REFERENCES batch_updates (`batch_id`, `update_id`) ON DELETE CASCADE;

SET foreign_key_checks = 0;
ALTER TABLE batches_inst_coll_staging DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`), ALGORITHM=INPLACE, LOCK=NONE;
ALTER TABLE batch_inst_coll_cancellable_resources DROP PRIMARY KEY, ADD PRIMARY KEY (`batch_id`, `update_id`, `inst_coll`, `token`), ALGORITH=INPLACE, LOCK=NONE;
SET foreign_key_checks = 1;

DROP TRIGGER IF EXISTS batches_before_insert;
DROP TRIGGER IF EXISTS batches_before_update;
DROP TRIGGER IF EXISTS batches_after_update;
DROP TRIGGER IF EXISTS batches_inst_coll_staging_before_insert;
DROP TRIGGER IF EXISTS batches_inst_coll_staging_before_update;
DROP TRIGGER IF EXISTS batch_inst_coll_cancellable_resources_before_insert;
DROP TRIGGER IF EXISTS batch_inst_coll_cancellable_resources_before_update;

ALTER TABLE batch_updates ADD FOREIGN KEY (`batch_id`) REFERENCES batches (`id`) ON DELETE CASCADE;

ALTER TABLE batches DROP COLUMN update_added, ALGORITHM=INPLACE, LOCK=NONE;

COMMIT;
