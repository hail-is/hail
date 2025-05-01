DELIMITER $$

DROP FUNCTION IF EXISTS is_batch_cancelled $$
CREATE FUNCTION is_batch_cancelled (batch_id BIGINT)
RETURNS BOOLEAN NOT DETERMINISTIC
RETURN EXISTS (
  SELECT 1 FROM job_groups_cancelled AS c
  WHERE c.id           = batch_id
    AND c.job_group_id = 0
) $$


DROP FUNCTION IF EXISTS is_job_group_cancelled $$
CREATE FUNCTION is_job_group_cancelled (batch_id BIGINT, job_group_id INT)
RETURNS BOOLEAN NOT DETERMINISTIC
RETURN EXISTS (
  SELECT 1
  FROM job_group_self_and_ancestors AS self
  INNER JOIN job_groups_cancelled   AS c
     ON self.batch_id    = c.id
    AND self.ancestor_id = c.job_group_id
  WHERE self.batch_id     = batch_id
    AND self.job_group_id = job_group_id
) $$


DROP FUNCTION IF EXISTS is_job_cancelled $$
CREATE FUNCTION is_job_cancelled (batch_id BIGINT, job_id INT)
RETURNS BOOLEAN NOT DETERMINISTIC
RETURN (
    SELECT NOT j.always_run AND (j.cancelled OR c.cancelled IS NOT NULL)
    FROM jobs AS j
    LEFT JOIN LATERAL (
      SELECT 1 AS cancelled
      FROM job_group_self_and_ancestors AS self
      INNER JOIN job_groups_cancelled   AS c
         ON self.batch_id    = c.id
        AND self.ancestor_id = c.job_group_id
      WHERE self.batch_id     = j.batch_id
        AND self.job_group_id = j.job_group_id
    ) AS c ON TRUE
    WHERE j.batch_id = batch_id
      AND j.job_id   = job_id
) $$


DROP TRIGGER IF EXISTS jobs_before_insert $$
CREATE TRIGGER jobs_before_insert BEFORE INSERT ON jobs
FOR EACH ROW
BEGIN
    IF is_job_group_cancelled(NEW.batch_id, NEW.job_group_id)
    THEN
      SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = "job group has already been cancelled";
    END IF;
END $$


DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_job_group_cancelled BOOLEAN;
  DECLARE cur_n_tokens INT;
  DECLARE rand_token INT;

  DECLARE always_run boolean;
  DECLARE cores_mcpu bigint;

  DECLARE was_marked_cancelled boolean;
  DECLARE was_cancelled        boolean;
  DECLARE was_cancellable      boolean;

  DECLARE now_marked_cancelled boolean;
  DECLARE now_cancelled        boolean;
  DECLARE now_cancellable      boolean;

  DECLARE was_ready boolean;
  DECLARE now_ready boolean;

  DECLARE was_running boolean;
  DECLARE now_running boolean;

  DECLARE was_creating boolean;
  DECLARE now_creating boolean;

  DECLARE delta_n_ready_cancellable_jobs          int;
  DECLARE delta_ready_cancellable_cores_mcpu   bigint;
  DECLARE delta_n_ready_jobs                      int;
  DECLARE delta_ready_cores_mcpu               bigint;
  DECLARE delta_n_cancelled_ready_jobs            int;

  DECLARE delta_n_running_cancellable_jobs        int;
  DECLARE delta_running_cancellable_cores_mcpu bigint;
  DECLARE delta_n_running_jobs                    int;
  DECLARE delta_running_cores_mcpu             bigint;
  DECLARE delta_n_cancelled_running_jobs          int;

  DECLARE delta_n_creating_cancellable_jobs       int;
  DECLARE delta_n_creating_jobs                   int;
  DECLARE delta_n_cancelled_creating_jobs         int;

  SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

  SELECT is_job_group_cancelled(OLD.batch_id, OLD.job_group_id)
  INTO cur_job_group_cancelled
  FOR SHARE;

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SET always_run = old.always_run; # always_run is immutable
  SET cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

  SET was_marked_cancelled = old.cancelled OR cur_job_group_cancelled;
  SET was_cancelled        = NOT always_run AND was_marked_cancelled;
  SET was_cancellable      = NOT always_run AND NOT was_marked_cancelled;

  SET now_marked_cancelled = new.cancelled or cur_job_group_cancelled;
  SET now_cancelled        = NOT always_run AND now_marked_cancelled;
  SET now_cancellable      = NOT always_run AND NOT now_marked_cancelled;

  # NB: was_cancelled => now_cancelled b/c you cannot be uncancelled

  SET was_ready    = old.state = 'Ready';
  SET now_ready    = new.state = 'Ready';
  SET was_running  = old.state = 'Running';
  SET now_running  = new.state = 'Running';
  SET was_creating = old.state = 'Creating';
  SET now_creating = new.state = 'Creating';

  SET delta_n_ready_cancellable_jobs        = (-1 * was_ready    *  was_cancellable  )     + (now_ready    *  now_cancellable  ) ;
  SET delta_n_ready_jobs                    = (-1 * was_ready    * (NOT was_cancelled))    + (now_ready    * (NOT now_cancelled));
  SET delta_n_cancelled_ready_jobs          = (-1 * was_ready    *  was_cancelled    )     + (now_ready    *  now_cancelled    ) ;

  SET delta_n_running_cancellable_jobs      = (-1 * was_running  *  was_cancellable  )     + (now_running  *  now_cancellable  ) ;
  SET delta_n_running_jobs                  = (-1 * was_running  * (NOT was_cancelled))    + (now_running  * (NOT now_cancelled));
  SET delta_n_cancelled_running_jobs        = (-1 * was_running  *  was_cancelled    )     + (now_running  *  now_cancelled    ) ;

  SET delta_n_creating_cancellable_jobs     = (-1 * was_creating *  was_cancellable  )     + (now_creating *  now_cancellable  ) ;
  SET delta_n_creating_jobs                 = (-1 * was_creating * (NOT was_cancelled))    + (now_creating * (NOT now_cancelled));
  SET delta_n_cancelled_creating_jobs       = (-1 * was_creating *  was_cancelled    )     + (now_creating *  now_cancelled    ) ;

  SET delta_ready_cancellable_cores_mcpu    = delta_n_ready_cancellable_jobs * cores_mcpu;
  SET delta_ready_cores_mcpu                = delta_n_ready_jobs * cores_mcpu;

  SET delta_running_cancellable_cores_mcpu  = delta_n_running_cancellable_jobs * cores_mcpu;
  SET delta_running_cores_mcpu              = delta_n_running_jobs * cores_mcpu;

  INSERT INTO job_group_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token,
    n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs,
    n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  SELECT NEW.batch_id, NEW.update_id, job_group_self_and_ancestors.ancestor_id, NEW.inst_coll, rand_token,
    delta_n_ready_cancellable_jobs,
    delta_ready_cancellable_cores_mcpu,
    delta_n_creating_cancellable_jobs,
    delta_n_running_cancellable_jobs,
    delta_running_cancellable_cores_mcpu
  FROM job_group_self_and_ancestors
  WHERE job_group_self_and_ancestors.batch_id = NEW.batch_id AND job_group_self_and_ancestors.job_group_id = NEW.job_group_id
  ON DUPLICATE KEY UPDATE
    n_ready_cancellable_jobs = n_ready_cancellable_jobs + delta_n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu + delta_ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs = n_creating_cancellable_jobs + delta_n_creating_cancellable_jobs,
    n_running_cancellable_jobs = n_running_cancellable_jobs + delta_n_running_cancellable_jobs,
    running_cancellable_cores_mcpu = running_cancellable_cores_mcpu + delta_running_cancellable_cores_mcpu;

  INSERT INTO user_inst_coll_resources (user, inst_coll, token,
    n_ready_jobs,
    n_running_jobs,
    n_creating_jobs,
    ready_cores_mcpu,
    running_cores_mcpu,
    n_cancelled_ready_jobs,
    n_cancelled_running_jobs,
    n_cancelled_creating_jobs
  )
  VALUES (cur_user, NEW.inst_coll, rand_token,
    delta_n_ready_jobs,
    delta_n_running_jobs,
    delta_n_creating_jobs,
    delta_ready_cores_mcpu,
    delta_running_cores_mcpu,
    delta_n_cancelled_ready_jobs,
    delta_n_cancelled_running_jobs,
    delta_n_cancelled_creating_jobs
  )
  ON DUPLICATE KEY UPDATE
    n_ready_jobs = n_ready_jobs + delta_n_ready_jobs,
    n_running_jobs = n_running_jobs + delta_n_running_jobs,
    n_creating_jobs = n_creating_jobs + delta_n_creating_jobs,
    ready_cores_mcpu = ready_cores_mcpu + delta_ready_cores_mcpu,
    running_cores_mcpu = running_cores_mcpu + delta_running_cores_mcpu,
    n_cancelled_ready_jobs = n_cancelled_ready_jobs + delta_n_cancelled_ready_jobs,
    n_cancelled_running_jobs = n_cancelled_running_jobs + delta_n_cancelled_running_jobs,
    n_cancelled_creating_jobs = n_cancelled_creating_jobs + delta_n_cancelled_creating_jobs;
END $$


DROP PROCEDURE IF EXISTS cancel_job_group $$
CREATE PROCEDURE cancel_job_group(
  IN in_batch_id VARCHAR(100),
  IN in_job_group_id INT
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_job_group_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_job_group_state
  FROM job_groups
  WHERE batch_id = in_batch_id AND job_group_id = in_job_group_id
  FOR UPDATE;

  SELECT is_job_group_cancelled(in_batch_id, in_job_group_id)
  INTO cur_cancelled
  FOR UPDATE;

  IF NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM job_group_inst_coll_cancellable_resources
    INNER JOIN batches ON job_group_inst_coll_cancellable_resources.batch_id = batches.id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND
      job_group_inst_coll_cancellable_resources.job_group_id = in_job_group_id AND
      batch_updates.committed
    GROUP BY user, inst_coll
    FOR UPDATE
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_creating_jobs = n_creating_jobs - @n_creating_cancellable_jobs,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs,
      n_cancelled_creating_jobs = n_cancelled_creating_jobs + @n_creating_cancellable_jobs;

    INSERT INTO job_group_inst_coll_cancellable_resources (batch_id, update_id, job_group_id, inst_coll, token,
      n_ready_cancellable_jobs,
      ready_cancellable_cores_mcpu,
      n_creating_cancellable_jobs,
      n_running_cancellable_jobs,
      running_cancellable_cores_mcpu)
    SELECT batch_id, update_id, ancestor_id, inst_coll, 0,
      -1 * (@jg_n_ready_cancellable_jobs := old_n_ready_cancellable_jobs),
      -1 * (@jg_ready_cancellable_cores_mcpu := old_ready_cancellable_cores_mcpu),
      -1 * (@jg_n_creating_cancellable_jobs := old_n_creating_cancellable_jobs),
      -1 * (@jg_n_running_cancellable_jobs := old_n_running_cancellable_jobs),
      -1 * (@jg_running_cancellable_cores_mcpu := old_running_cancellable_cores_mcpu)
    FROM job_group_self_and_ancestors
    INNER JOIN LATERAL (
      SELECT update_id, inst_coll, COALESCE(SUM(n_ready_cancellable_jobs), 0) AS old_n_ready_cancellable_jobs,
        COALESCE(SUM(ready_cancellable_cores_mcpu), 0) AS old_ready_cancellable_cores_mcpu,
        COALESCE(SUM(n_creating_cancellable_jobs), 0) AS old_n_creating_cancellable_jobs,
        COALESCE(SUM(n_running_cancellable_jobs), 0) AS old_n_running_cancellable_jobs,
        COALESCE(SUM(running_cancellable_cores_mcpu), 0) AS old_running_cancellable_cores_mcpu
      FROM job_group_inst_coll_cancellable_resources
      WHERE job_group_inst_coll_cancellable_resources.batch_id = job_group_self_and_ancestors.batch_id AND
        job_group_inst_coll_cancellable_resources.job_group_id = job_group_self_and_ancestors.job_group_id
      GROUP BY update_id, inst_coll
      FOR UPDATE
    ) AS t ON TRUE
    WHERE job_group_self_and_ancestors.batch_id = in_batch_id AND job_group_self_and_ancestors.job_group_id = in_job_group_id
    ON DUPLICATE KEY UPDATE
      n_ready_cancellable_jobs = n_ready_cancellable_jobs - @jg_n_ready_cancellable_jobs,
      ready_cancellable_cores_mcpu = ready_cancellable_cores_mcpu - @jg_ready_cancellable_cores_mcpu,
      n_creating_cancellable_jobs = n_creating_cancellable_jobs - @jg_n_creating_cancellable_jobs,
      n_running_cancellable_jobs = n_running_cancellable_jobs - @jg_n_running_cancellable_jobs,
      running_cancellable_cores_mcpu = running_cancellable_cores_mcpu - @jg_running_cancellable_cores_mcpu;

    # Group cancellation, like any operation, must be O(1) time. The number of descendant groups is unbounded,
    # so we neither delete rows from job_group_inst_coll_cancellable_resources nor update job_groups_cancelled.
    # The former is handled by main.py. In the latter case, group cancellation state is implicitly defined by an
    # upwards traversal on the ancestor tree.

    INSERT INTO job_groups_cancelled (id, job_group_id)
    VALUES (in_batch_id, in_job_group_id);
  END IF;

  COMMIT;
END $$


DROP PROCEDURE IF EXISTS cancel_batch $$
CREATE PROCEDURE cancel_batch(
  IN in_batch_id VARCHAR(100)
)
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_state VARCHAR(40);
  DECLARE cur_cancelled BOOLEAN;
  DECLARE cur_n_cancelled_ready_jobs INT;
  DECLARE cur_cancelled_ready_cores_mcpu BIGINT;
  DECLARE cur_n_cancelled_running_jobs INT;
  DECLARE cur_cancelled_running_cores_mcpu BIGINT;
  DECLARE cur_n_n_cancelled_creating_jobs INT;

  START TRANSACTION;

  SELECT user, `state` INTO cur_user, cur_batch_state FROM batches
  WHERE id = in_batch_id
  FOR UPDATE;

  SELECT is_batch_cancelled(in_batch_id)
  INTO cur_cancelled
  FOR UPDATE;

  IF cur_batch_state = 'running' AND NOT cur_cancelled THEN
    INSERT INTO user_inst_coll_resources (user, inst_coll, token,
      n_ready_jobs, ready_cores_mcpu,
      n_running_jobs, running_cores_mcpu,
      n_creating_jobs,
      n_cancelled_ready_jobs, n_cancelled_running_jobs, n_cancelled_creating_jobs)
    SELECT user, inst_coll, 0,
      -1 * (@n_ready_cancellable_jobs := COALESCE(SUM(n_ready_cancellable_jobs), 0)),
      -1 * (@ready_cancellable_cores_mcpu := COALESCE(SUM(ready_cancellable_cores_mcpu), 0)),
      -1 * (@n_running_cancellable_jobs := COALESCE(SUM(n_running_cancellable_jobs), 0)),
      -1 * (@running_cancellable_cores_mcpu := COALESCE(SUM(running_cancellable_cores_mcpu), 0)),
      -1 * (@n_creating_cancellable_jobs := COALESCE(SUM(n_creating_cancellable_jobs), 0)),
      COALESCE(SUM(n_ready_cancellable_jobs), 0),
      COALESCE(SUM(n_running_cancellable_jobs), 0),
      COALESCE(SUM(n_creating_cancellable_jobs), 0)
    FROM job_group_inst_coll_cancellable_resources
    JOIN batches ON batches.id = job_group_inst_coll_cancellable_resources.batch_id
    INNER JOIN batch_updates ON job_group_inst_coll_cancellable_resources.batch_id = batch_updates.batch_id AND
      job_group_inst_coll_cancellable_resources.update_id = batch_updates.update_id
    WHERE job_group_inst_coll_cancellable_resources.batch_id = in_batch_id AND batch_updates.committed
    GROUP BY user, inst_coll
    ON DUPLICATE KEY UPDATE
      n_ready_jobs = n_ready_jobs - @n_ready_cancellable_jobs,
      ready_cores_mcpu = ready_cores_mcpu - @ready_cancellable_cores_mcpu,
      n_running_jobs = n_running_jobs - @n_running_cancellable_jobs,
      running_cores_mcpu = running_cores_mcpu - @running_cancellable_cores_mcpu,
      n_creating_jobs = n_creating_jobs - @n_creating_cancellable_jobs,
      n_cancelled_ready_jobs = n_cancelled_ready_jobs + @n_ready_cancellable_jobs,
      n_cancelled_running_jobs = n_cancelled_running_jobs + @n_running_cancellable_jobs,
      n_cancelled_creating_jobs = n_cancelled_creating_jobs + @n_creating_cancellable_jobs;

    # there are no cancellable jobs left, they have been cancelled
    DELETE FROM job_group_inst_coll_cancellable_resources WHERE batch_id = in_batch_id;

    # cancel root job group only
    INSERT INTO job_groups_cancelled (id, job_group_id) VALUES (in_batch_id, 0);
  END IF;

  COMMIT;
END $$


DROP PROCEDURE IF EXISTS schedule_job $$
CREATE PROCEDURE schedule_job(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100)
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE cur_attempt_id VARCHAR(40);
  DECLARE delta_cores_mcpu INT;
  DECLARE cur_instance_is_pool BOOLEAN;

  START TRANSACTION;

  SELECT state, cores_mcpu, attempt_id
  INTO cur_job_state, cur_cores_mcpu, cur_attempt_id
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT is_job_cancelled(in_batch_id, in_job_id)
  INTO cur_job_cancel
  FOR SHARE;

  SELECT is_pool
  INTO cur_instance_is_pool
  FROM instances
  LEFT JOIN inst_colls ON instances.inst_coll = inst_colls.name
  WHERE instances.name = in_instance_name;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  IF cur_instance_is_pool THEN
    IF delta_cores_mcpu = 0 THEN
      SET delta_cores_mcpu = cur_cores_mcpu;
    ELSE
      SET delta_cores_mcpu = 0;
    END IF;
  END IF;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF (cur_job_state = 'Ready' OR cur_job_state = 'Creating') AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
    COMMIT;
    SELECT 0 as rc, in_instance_name, delta_cores_mcpu;
  ELSE
    COMMIT;
    SELECT 1 as rc,
      cur_job_state,
      cur_job_cancel,
      cur_instance_state,
      in_instance_name,
      cur_attempt_id,
      delta_cores_mcpu,
      'job not Ready or cancelled or instance not active, but attempt already exists' as message;
  END IF;
END $$


DROP PROCEDURE IF EXISTS mark_job_creating $$
CREATE PROCEDURE mark_job_creating(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT is_job_cancelled(in_batch_id, in_job_id)
  INTO cur_job_cancel
  FOR SHARE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'pending' THEN
    UPDATE jobs SET state = 'Creating', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$


DROP PROCEDURE IF EXISTS mark_job_started $$
CREATE PROCEDURE mark_job_started(
  IN in_batch_id BIGINT,
  IN in_job_id INT,
  IN in_attempt_id VARCHAR(40),
  IN in_instance_name VARCHAR(100),
  IN new_start_time BIGINT
)
BEGIN
  DECLARE cur_job_state VARCHAR(40);
  DECLARE cur_job_cancel BOOLEAN;
  DECLARE cur_cores_mcpu INT;
  DECLARE cur_instance_state VARCHAR(40);
  DECLARE delta_cores_mcpu INT;

  START TRANSACTION;

  SELECT state, cores_mcpu
  INTO cur_job_state, cur_cores_mcpu
  FROM jobs
  WHERE batch_id = in_batch_id AND job_id = in_job_id
  FOR UPDATE;

  SELECT is_job_cancelled(in_batch_id, in_job_id)
  INTO cur_job_cancel
  FOR SHARE;

  CALL add_attempt(in_batch_id, in_job_id, in_attempt_id, in_instance_name, cur_cores_mcpu, delta_cores_mcpu);

  UPDATE attempts SET start_time = new_start_time, rollup_time = new_start_time
  WHERE batch_id = in_batch_id AND job_id = in_job_id AND attempt_id = in_attempt_id;

  SELECT state INTO cur_instance_state FROM instances WHERE name = in_instance_name LOCK IN SHARE MODE;

  IF cur_job_state = 'Ready' AND NOT cur_job_cancel AND cur_instance_state = 'active' THEN
    UPDATE jobs SET state = 'Running', attempt_id = in_attempt_id WHERE batch_id = in_batch_id AND job_id = in_job_id;
  END IF;

  COMMIT;
  SELECT 0 as rc, delta_cores_mcpu;
END $$


DELIMITER ;
