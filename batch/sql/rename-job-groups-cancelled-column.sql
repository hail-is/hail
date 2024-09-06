/*
mysql> SELECT * FROM INFORMATION_SCHEMA.INNODB_SYS_FOREIGN \G

(mysql 8.x or above)
mysql> SELECT * FROM INFORMATION_SCHEMA.INNODB_FOREIGN \G 

*************************** XXX row ***************************
      ID: batches/job_groups_cancelled_ibfk_1
FOR_NAME: batches/job_groups_cancelled
REF_NAME: batches/batches
  N_COLS: 1
    TYPE: 33
*************************** YYY row ***************************
      ID: batches/job_groups_cancelled_ibfk_2
FOR_NAME: batches/job_groups_cancelled
REF_NAME: batches/job_groups
  N_COLS: 2
    TYPE: 33
*/

ALTER TABLE job_groups_cancelled DROP FOREIGN KEY job_groups_cancelled_ibfk_1;
ALTER TABLE job_groups_cancelled DROP FOREIGN KEY job_groups_cancelled_ibfk_2;
ALTER TABLE job_groups_cancelled DROP PRIMARY KEY;
ALTER TABLE job_groups_cancelled CHANGE COLUMN `id` `batch_id` BIGINT NOT NULL;
ALTER TABLE job_groups_cancelled ADD PRIMARY KEY (`batch_id`, `job_group_id`),
  ADD FOREIGN KEY (`batch_id`) REFERENCES batches(id) ON DELETE CASCADE,
  ADD FOREIGN KEY (`batch_id`, `job_group_id`) REFERENCES job_groups (`batch_id`, `job_group_id`) ON DELETE CASCADE;
