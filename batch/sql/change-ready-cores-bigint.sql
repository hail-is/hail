ALTER TABLE ready_cores MODIFY ready_cores_mcpu BIGINT;

ALTER TABLE user_resources MODIFY ready_cores_mcpu BIGINT;
ALTER TABLE user_resources MODIFY running_cores_mcpu BIGINT;
