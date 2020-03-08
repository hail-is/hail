CREATE TABLE authorized_shas (
  sha VARCHAR(100) NOT NULL
) ENGINE = InnoDB;

CREATE INDEX authorized_shas_sha ON authorized_shas (sha);
