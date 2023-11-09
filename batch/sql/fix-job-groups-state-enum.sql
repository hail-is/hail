DELIMITER $$

DROP TRIGGER IF EXISTS batches_after_update $$
CREATE TRIGGER batches_after_update AFTER UPDATE ON batches
FOR EACH ROW
BEGIN
  DECLARE jg_state VARCHAR(40);

  SET jg_state = IF(NEW.state = "open", "complete", NEW.state);

  IF OLD.migrated_batch = 0 AND NEW.migrated_batch = 1 THEN
    INSERT INTO job_groups (batch_id, job_group_id, `user`, cancel_after_n_failures, `state`, n_jobs, time_created, time_completed, callback, attributes)
    VALUES (NEW.id, 0, NEW.`user`, NEW.cancel_after_n_failures, jg_state, NEW.n_jobs, NEW.time_created, NEW.time_completed, NEW.callback, NEW.attributes);

    INSERT INTO job_group_self_and_ancestors (batch_id, job_group_id, ancestor_id, `level`)
    VALUES (NEW.id, 0, 0, 0);
  END IF;
END $$

DELIMITER ;
