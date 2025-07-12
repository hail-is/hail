/*
mysql> SELECT * FROM INFORMATION_SCHEMA.INNODB_SYS_FOREIGN \G

      ID: batch/batches_ibfk_1
FOR_NAME: batch/batches
REF_NAME: batch/user_resources
  N_COLS: 1
    TYPE: 1
*/

ALTER TABLE batches DROP FOREIGN KEY batches_ibfk_1;
