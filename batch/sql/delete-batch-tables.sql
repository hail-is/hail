DROP PROCEDURE IF EXISTS recompute_incremental;
DROP PROCEDURE IF EXISTS cancel_batch;
DROP PROCEDURE IF EXISTS activate_instance;
DROP PROCEDURE IF EXISTS deactivate_instance;
DROP PROCEDURE IF EXISTS mark_instance_deleted;
DROP PROCEDURE IF EXISTS close_batch;
DROP PROCEDURE IF EXISTS schedule_job;
DROP PROCEDURE IF EXISTS unschedule_job;
DROP PROCEDURE IF EXISTS mark_job_creating;
DROP PROCEDURE IF EXISTS mark_job_started;
DROP PROCEDURE IF EXISTS mark_job_complete;
DROP PROCEDURE IF EXISTS add_attempt;

DROP TRIGGER IF EXISTS instances_before_update;
DROP TRIGGER IF EXISTS attempts_before_update;
DROP TRIGGER IF EXISTS attempts_after_update;
DROP TRIGGER IF EXISTS jobs_after_update;
DROP TRIGGER IF EXISTS attempt_resources_after_insert;

DROP TABLE IF EXISTS `aggregated_billing_project_resources`;
DROP TABLE IF EXISTS `aggregated_batch_resources`;
DROP TABLE IF EXISTS `aggregated_job_resources`;
DROP TABLE IF EXISTS `attempt_resources`;
DROP TABLE IF EXISTS `batch_cancellable_resources`;  # deprecated
DROP TABLE IF EXISTS `batch_inst_coll_cancellable_resources`;
DROP TABLE IF EXISTS `globals`;
DROP TABLE IF EXISTS `attempts`;
DROP TABLE IF EXISTS `batch_attributes`;
DROP TABLE IF EXISTS `job_attributes`;
DROP TABLE IF EXISTS `job_parents`;
DROP TABLE IF EXISTS `batch_bunches`;
DROP TABLE IF EXISTS `ready_cores`;  # deprecated
DROP TABLE IF EXISTS `gevents_mark`;
DROP TABLE IF EXISTS `jobs`;
DROP TABLE IF EXISTS `batches_cancelled`;
DROP TABLE IF EXISTS `batches_staging`;  # deprecated
DROP TABLE IF EXISTS `batches_inst_coll_staging`;
DROP TABLE IF EXISTS `batches`;
DROP TABLE IF EXISTS `user_resources`;  # deprecated
DROP TABLE IF EXISTS `user_inst_coll_resources`;
DROP TABLE IF EXISTS `instances_free_cores_mcpu`;
DROP TABLE IF EXISTS `instances`;
DROP TABLE IF EXISTS `billing_project_users`;
DROP TABLE IF EXISTS `billing_projects`;
DROP TABLE IF EXISTS `batch_migration_version`;
DROP TABLE IF EXISTS `batch_migrations`;
DROP TABLE IF EXISTS `pools`;
DROP TABLE IF EXISTS `inst_colls`;
DROP TABLE IF EXISTS `resources`;
