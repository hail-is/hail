CREATE TABLE IF NOT EXISTS `regions` (
  `region_id` INT NOT NULL AUTO_INCREMENT,
  `region` VARCHAR(40) NOT NULL,
  PRIMARY KEY (`region_id`),
  UNIQUE(region)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `job_regions` (
  `batch_id` BIGINT NOT NULL,
  `job_id` INT NOT NULL,
  `region_id` INT NOT NULL,
  PRIMARY KEY (`batch_id`, `job_id`, `region_id`),
  FOREIGN KEY (`batch_id`, `job_id`) REFERENCES jobs(batch_id, job_id) ON DELETE CASCADE,
  FOREIGN KEY (`region_id`) REFERENCES regions(region_id) ON DELETE CASCADE
) ENGINE = InnoDB;

ALTER TABLE jobs ADD COLUMN n_regions INT DEFAULT NULL, ALGORITHM=INSTANT;
ALTER TABLE jobs ADD COLUMN regions_bits_rep BIGINT DEFAULT NULL, ALGORITHM=INSTANT;

CREATE INDEX `jobs_batch_id_always_run_n_regions_regions_bits_rep_job_id` ON `jobs` (`batch_id`, `always_run`, `n_regions`, `regions_bits_rep`, `job_id`);
