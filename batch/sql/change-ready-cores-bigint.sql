ALTER TABLE ready_cores MODIFY ready_cores_mcpu BIGINT NOT NULL;

ALTER TABLE user_resources MODIFY ready_cores_mcpu BIGINT NOT NULL DEFAULT 0;
ALTER TABLE user_resources MODIFY running_cores_mcpu BIGINT NOT NULL DEFAULT 0;
