CREATE TABLE authorized_shas (
  sha VARCHAR(100) NOT NULL
) ENGINE = InnoDB;

CREATE INDEX authorized_shas_sha ON authorized_shas (sha);

CREATE TABLE alerted_failed_shas (
  sha VARCHAR(100) NOT NULL,
  PRIMARY KEY (`sha`)
) ENGINE = InnoDB;

CREATE TABLE invalidated_batches (
  batch_id BIGINT NOT NULL
) ENGINE = InnoDB;

CREATE INDEX invalidated_batches_batch_id ON invalidated_batches (batch_id);

CREATE TABLE IF NOT EXISTS `globals` (
  `frozen_merge_deploy` BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE = InnoDB;

INSERT INTO `globals` (frozen_merge_deploy) VALUES (FALSE);

CREATE TABLE IF NOT EXISTS `active_namespaces` (
  `namespace` VARCHAR(100) NOT NULL,
  `creation_time` TIMESTAMP NOT NULL DEFAULT (UTC_TIMESTAMP),
  `expiration_time` TIMESTAMP,
  PRIMARY KEY (`namespace`)
) ENGINE = InnoDB;

CREATE TABLE IF NOT EXISTS `deployed_services` (
  `namespace` VARCHAR(100) NOT NULL,
  `service` VARCHAR(100) NOT NULL,
  `rate_limit_rps` INT,
  PRIMARY KEY (`namespace`, `service`),
  FOREIGN KEY (`namespace`) REFERENCES active_namespaces(namespace) ON DELETE CASCADE
) ENGINE = InnoDB;

INSERT INTO `active_namespaces` (`namespace`) VALUES (`default`);
INSERT INTO `deployed_services` (`namespace`, `service`) VALUES
('default', 'auth'), ('default', 'batch'), ('default', 'batch-driver'), ('default', 'ci');

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
