DELIMITER $$

DROP TRIGGER IF EXISTS jobs_after_update $$
CREATE TRIGGER jobs_after_update AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN

DECLARE cur_user VARCHAR(100);
DECLARE cur_batch_cancelled BOOLEAN;
DECLARE cur_n_tokens INT;
DECLARE rand_token INT;

SELECT user INTO cur_user FROM batches WHERE id = NEW.batch_id;

SET cur_batch_cancelled = EXISTS (SELECT TRUE
                                  FROM batches_cancelled
                                  WHERE id = NEW.batch_id
                                  LOCK IN SHARE MODE);

SELECT n_tokens INTO cur_n_tokens FROM globals LOCK IN SHARE MODE;
SET rand_token = FLOOR(RAND() * cur_n_tokens);

# NB: we cannot write something sane like
#
#     let x = y;
#
# so we instead write
#
#     let x type not null default y;
#
declare always_run boolean not null default old.always_run; # always_run is immutable
declare cores_mcpu boolean not null default old.cores_mcpu; # cores_mcpu is immutable

declare was_marked_cancelled boolean not null default old.cancelled or cur_batch_cancelled;
declare was_cancelled        boolean not null default not always_run and was_marked_cancelled;
declare was_cancellable      boolean not null default not always_run and not was_marked_cancelled;

declare now_marked_cancelled boolean not null default new.cancelled or cur_batch_cancelled;
declare now_cancelled        boolean not null default not always_run and now_marked_cancelled;
declare now_cancellable      boolean not null default not always_run and not now_marked_cancelled;

# NB: was_cancelled => now_cancelled b/c you cannot be uncancelled

declare was_ready boolean not null default old.state = 'Ready';
declare now_ready boolean not null default new.state = 'Ready';
declare was_running boolean not null default old.state = 'Running';
declare now_running boolean not null default new.state = 'Running';
declare was_creating boolean not null default old.state = 'Creating';
declare now_creating boolean not null default new.state = 'Creating';

declare delta_n_ready_cancellable_jobs          int not null default  (-1 * was_ready    * was_cancellable  ) + (now_ready    * now_cancellable  );
declare delta_ready_cancellable_cores_mcpu   bigint not null default ((-1 * was_ready    * was_cancellable  ) + (now_ready    * now_cancellable  )) * cores_mcpu;
declare delta_n_ready_jobs                      int not null default  (-1 * was_ready    * not was_cancelled) + (now_ready    * not now_cancelled);
declare delta_ready_cores_mcpu               bigint not null default ((-1 * was_ready    * not was_cancelled) + (now_ready    * not now_cancelled)) * cores_mcpu;
declare delta_n_cancelled_ready_jobs            int not null default  (-1 * was_ready    * was_cancelled    ) + (now_ready    * now_cancelled    );

declare delta_n_running_cancellable_jobs        int not null default  (-1 * was_running  * was_cancellable  ) + (now_running  * now_cancellable  );
declare delta_running_cancellable_cores_mcpu bigint not null default ((-1 * was_running  * was_cancellable  ) + (now_running  * now_cancellable  )) * cores_mcpu;
declare delta_n_running_jobs                    int not null default  (-1 * was_running  * not was_cancelled) + (now_running  * not now_cancelled);
declare delta_running_cores_mcpu             bigint not null default ((-1 * was_running  * not was_cancelled) + (now_running  * not now_cancelled)) * cores_mcpu;
declare delta_n_cancelled_running_jobs          int not null default  (-1 * was_running  * was_cancelled    ) + (now_running  * now_cancelled    );

declare delta_n_creating_cancellable_jobs       int not null default  (-1 * was_creating * was_cancellable  ) + (now_creating * now_cancellable  );
declare delta_n_creating_jobs                   int not null default  (-1 * was_creating * not was_cancelled) + (now_creating * not now_cancelled);
declare delta_n_cancelled_creating_jobs         int not null default  (-1 * was_creating * was_cancelled    ) + (now_creating * now_cancelled    );

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
