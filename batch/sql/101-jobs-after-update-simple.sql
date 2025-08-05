DELIMITER $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  DECLARE cur_user VARCHAR(100);
  DECLARE cur_batch_cancelled BOOLEAN;
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

  SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                    FROM batches_cancelled
                                    WHERE id = NEW.batch_id
                                    LOCK IN SHARE MODE);

  SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
  SET rand_token = FLOOR(RAND() * cur_n_tokens);

  SET always_run = old.always_run; # always_run is immutable
  SET cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

  SET was_marked_cancelled = old.cancelled OR cur_batch_cancelled;
  SET was_cancelled        = NOT always_run AND was_marked_cancelled;
  SET was_cancellable      = NOT always_run AND NOT was_marked_cancelled;

  SET now_marked_cancelled = new.cancelled or cur_batch_cancelled;
  SET now_cancelled        = NOT always_run AND now_marked_cancelled;
  SET now_cancellable      = NOT always_run AND NOT now_marked_cancelled;

  # NB: was_cancelled implies now_cancelled b/c you cannot be uncancelled

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

  INSERT INTO batch_inst_coll_cancellable_resources (batch_id, update_id, inst_coll, token,
    n_ready_cancellable_jobs,
    ready_cancellable_cores_mcpu,
    n_creating_cancellable_jobs,
    n_running_cancellable_jobs,
    running_cancellable_cores_mcpu)
  VALUES (NEW.batch_id, NEW.update_id, NEW.inst_coll, rand_token,
    delta_n_ready_cancellable_jobs,
    delta_ready_cancellable_cores_mcpu,
    delta_n_creating_cancellable_jobs,
    delta_n_running_cancellable_jobs,
    delta_running_cancellable_cores_mcpu)
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

DELIMITER ;
