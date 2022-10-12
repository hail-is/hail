CREATE INDEX `jobs_batch_id_always_run_n_regions_regions_bits_rep_int_job_id` ON `jobs` (`batch_id`, `always_run`, `n_regions`, `regions_bits_rep_int`, `job_id`);
CREATE INDEX `jobs_batch_id_n_regions_regions_bits_rep_int_job_id` ON `jobs` (`batch_id`, `n_regions`, `regions_bits_rep_int`, `job_id`);
