CREATE TABLE authorized_shas (
  sha VARCHAR(100) NOT NULL
) ENGINE = InnoDB;

CREATE INDEX authorized_shas_sha ON authorized_shas (sha);

CREATE TABLE invalidated_batches (
  batch_id BIGINT NOT NULL
) ENGINE = InnoDB;

CREATE INDEX invalidated_batches_batch_id ON invalidated_batches (batch_id);

CREATE TABLE IF NOT EXISTS `globals` (
  `frozen_merge_deploy` BOOLEAN NOT NULL DEFAULT FALSE
) ENGINE = InnoDB;

INSERT INTO `globals` (frozen_merge_deploy) VALUES (FALSE);
