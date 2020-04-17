CREATE TABLE invalidated_batches (
  batch_id BIGINT NOT NULL
) ENGINE = InnoDB;

CREATE INDEX invalidated_batches_batch_id ON invalidated_batches (batch_id);
