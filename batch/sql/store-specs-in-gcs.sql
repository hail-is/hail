ALTER TABLE batches ADD COLUMN format_version INT NOT NULL DEFAULT 1;
ALTER TABLE batches MODIFY COLUMN format_version INT NOT NULL;

CREATE TABLE IF NOT EXISTS `batch_bunches` (
  `batch_id` BIGINT NOT NULL,
  `token` VARCHAR(100) NOT NULL,
  `start_job_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `token`),
  FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  FOREIGN KEY (`start_job_id`) REFERENCES jobs(job_id) ON DELETE CASCADE
) ENGINE = InnoDB;
CREATE INDEX `batch_bunches_start_job_id` ON `batch_bunches` (`batch_id`, `start_job_id`);

ALTER TABLE instances ADD COLUMN version INT NOT NULL DEFAULT 1;
ALTER TABLE instances MODIFY COLUMN version INT NOT NULL;
