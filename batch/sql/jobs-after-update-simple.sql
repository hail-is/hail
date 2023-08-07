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

set always_run = old.always_run; # always_run is immutable
set cores_mcpu = old.cores_mcpu; # cores_mcpu is immutable

set was_marked_cancelled = old.cancelled or cur_batch_cancelled;
set was_cancelled        = not always_run and was_marked_cancelled;
set was_cancellable      = not always_run and not was_marked_cancelled;

set now_marked_cancelled = new.cancelled or cur_batch_cancelled;
set now_cancelled        = not always_run and now_marked_cancelled;
set now_cancellable      = not always_run and not now_marked_cancelled;

# NB: was_cancelled => now_cancelled b/c you cannot be uncancelled

set was_ready    = old.state = 'Ready';
set now_ready    = new.state = 'Ready';
set was_running  = old.state = 'Running';
set now_running  = new.state = 'Running';
set was_creating = old.state = 'Creating';
set now_creating = new.state = 'Creating';

set delta_n_ready_cancellable_jobs              =  (-1      * was_ready    *  was_cancellable  )     + (now_ready    *  now_cancellable  )               ;
set delta_ready_cancellable_cores_mcpu          = ((-1      * was_ready    *  was_cancellable  )     + (now_ready    *  now_cancellable  ))  * cores_mcpu;
set delta_n_ready_jobs                          =  (-1      * was_ready    * (not was_cancelled))    + (now_ready    * (not now_cancelled))              ;
set delta_ready_cores_mcpu                      = ((-1      * was_ready    * (not was_cancelled))    + (now_ready    * (not now_cancelled))) * cores_mcpu;
set delta_n_cancelled_ready_jobs                =  (-1      * was_ready    *  was_cancelled    )     + (now_ready    *  now_cancelled    )               ;

set delta_n_running_cancellable_jobs            =  (-1      * was_running  *  was_cancellable  )     + (now_running  *  now_cancellable  )               ;
set delta_running_cancellable_cores_mcpu        = ((-1      * was_running  *  was_cancellable  )     + (now_running  *  now_cancellable  ))  * cores_mcpu;
set delta_n_running_jobs                        =  (-1      * was_running  * (not was_cancelled))    + (now_running  * (not now_cancelled))              ;
set delta_running_cores_mcpu                    = ((-1      * was_running  * (not was_cancelled))    + (now_running  * (not now_cancelled))) * cores_mcpu;
set delta_n_cancelled_running_jobs              =  (-1      * was_running  *  was_cancelled    )     + (now_running  *  now_cancelled    )               ;

set delta_n_creating_cancellable_jobs           =  (-1      * was_creating *  was_cancellable  )     + (now_creating *  now_cancellable  )               ;
set delta_n_creating_jobs                       =  (-1      * was_creating * (not was_cancelled))    + (now_creating * (not now_cancelled))              ;
set delta_n_cancelled_creating_jobs             =  (-1      * was_creating *  was_cancelled    )     + (now_creating *  now_cancelled    )               ;

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
