/* @param cb: should be standard node.js callback, err, $successResult */
const log = require.main.require("./common/logger");

module.exports = {
  start
};

// Client should implement .maxReserves property unless want zombie jobs
function start(beanstalkClient, maxReserves, cb) {
  // console.info('maxReserves', maxReserves);
  beanstalkClient.reserve_with_timeout(10, (err, jobID, payload) => {
    if (err === "TIMED_OUT") {
      return start(beanstalkClient, maxReserves, cb);
    }

    // Our most common error is with an undefiend queueID, and TIMED_OUT or
    // NOT_FOUND err
    // Not sure why; possibly this is normal behavior?
    if (err && jobID) {
      if (err !== "DEADLINE_SOON") {
        log.warn(`Deadline Soon for job ${jobID}: ${err}`);

        beanstalkClient.destroy(jobID, destroyErr => {
          // Our most common error is with an undefiend queueID...
          if (destroyErr && jobID) {
            log.error(`error destroying job with id ${jobID}: ${destroyErr}`);
          } else {
            log.debug(`destroyed job with id ${jobID} because it exceeded
              ${maxReserves} reservations`);
          }
        });
      }

      return start(beanstalkClient, maxReserves, cb);
    }

    let job = null;
    let parseError = null;

    try {
      job = JSON.parse(payload);
    } catch (e) {
      log.error(e, payload);
      parseError = e;
    }

    // We could bury these, but it's a bit pointless, since these jobs contain
    // no usable information, unless we've used a different encoding strategy
    // in which case we should figure this out early and solve
    if (parseError) {
      beanstalkClient.destroy(jobID, destroyErr => {
        if (err) {
          log.error(`error destroying job with id ${jobID}: ${destroyErr}`);
        } else {
          log.debug(`destroyed job with id ${jobID} due to: ${parseError}`);
        }

        // Regardless of the outcome of destroy, the callback needs to know
        // that their job failed to parse
        cb(parseError, null, null);

        return start(beanstalkClient, maxReserves, cb);
      });
    }

    // Do something with the data, and if nothing possible
    // cb must call a callback as a third argument
    // upon the success of whatever action the consumer implemented
    // we delete the job from the queue, ensuring that messages will be consumed

    cb(null, job, cbErr => {
      if (cbErr) {
        beanstalkClient.stats_job(jobID, (statErr, response) => {
          // TODO: what should we do if error on stats?
          if (statErr) {
            log.error(
              `error getting stats for job with id ${jobID} because ${statErr}`
            );

            return;
          }

          if (response.reserves > maxReserves) {
            beanstalkClient.destroy(jobID, dErr => {
              if (dErr) {
                log.error(
                  `error destroying job with id ${jobID}: ${response}, ${payload}, ${dErr}`
                );
              } else {
                log.debug(`destroyed job with id ${jobID} because it exceeded
                  ${maxReserves} reservations`);
              }
            });

            return;
          }
          // If consumer failed to do something useful with this job,
          // and the job hasn't exceeded max reservations, give a delayed attempt
          beanstalkClient.release(jobID, 0, 100, rErr => {
            if (rErr) {
              log.error(`failed to release job with id ${jobID}`);
            }
          });
        });

        return start(beanstalkClient, maxReserves, cb);
      }

      // The callback succeeded, job is useless now
      beanstalkClient.destroy(jobID, dErr => {
        if (dErr) {
          log.error(`failed to delete job with id ${jobID}`);
        }
      });

      return start(beanstalkClient, maxReserves, cb);
    });
  });
}
