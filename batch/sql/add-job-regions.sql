CREATE INDEX `jobs_batch_id_always_run_job_id` ON `jobs` (`batch_id`, `always_run`, `job_id`);

CREATE TABLE IF NOT EXISTS `region_ids` (
  `region_id` INT NOT NULL AUTO_INCREMENT,
  `region` VARCHAR(40) NOT NULL,
  PRIMARY KEY (`region_id`),
  UNIQUE(region)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `job_regions` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `region_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`),
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE,
  FOREIGN KEY (`region_id`) REFERENCES region_ids(region_id) ON DELETE CASCADE
) ENGINE = InnoDB;
