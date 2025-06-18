ALTER TABLE pools ADD COLUMN min_instances BIGINT NOT NULL DEFAULT 0;
UPDATE pools SET min_instances = CAST(enable_standing_worker=1 AS SIGNED INTEGER);
