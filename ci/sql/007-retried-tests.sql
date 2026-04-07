CREATE TABLE retried_tests (
  id BIGINT NOT NULL AUTO_INCREMENT,
  batch_id BIGINT NOT NULL,
  job_id INT NOT NULL,
  job_name VARCHAR(255),
  state VARCHAR(50) NOT NULL,
  exit_code INT,
  pr_number INT NOT NULL,
  target_branch VARCHAR(255) NOT NULL,
  source_branch VARCHAR(255) NOT NULL,
  source_sha VARCHAR(40) NOT NULL,
  retried_by VARCHAR(255) NOT NULL,
  retried_at TIMESTAMP NOT NULL DEFAULT (UTC_TIMESTAMP),
  PRIMARY KEY (id),
  INDEX retried_tests_batch_id (batch_id),
  INDEX retried_tests_pr_number (pr_number),
  INDEX retried_tests_job_name (job_name)
) ENGINE = InnoDB;
