ALTER TABLE jobs ADD COLUMN regions_bits_rep_int BIGINT DEFAULT NULL, ALGORITHM=INSTANT;

CREATE INDEX `jobs_always_run_n_regions_regions_bits_rep_int_batch_id_job_id` ON `jobs` (`always_run`, `n_regions`, `regions_bits_rep_int`, `batch_id`, `job_id`);
CREATE INDEX `jobs_n_regions_regions_bits_rep_int_batch_id_job_id` ON `jobs` (`n_regions`, `regions_bits_rep_int`, `batch_id`, `job_id`);
