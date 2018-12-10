const CronJob = require("cron").CronJob;
const moment = require("moment");
const log = require.main.require("./common/logger");

class JobFixer {
  /* @param {Obj} config Job config
     * @param {Obj} comm Job Comm instance
     * @param {Obj} Model Mongoose model
     */
  constructor(Jobs) {
    this.Jobs = Jobs;

    this.re = new RegExp("annotation|saveFromQuery");
  }

  run() {
    // run every day at 8am
    var job = new CronJob(
      "1 * * * * *",
      this.fix.bind(this),
      null,
      true,
      "America/New_York"
    );
    console.info("running");
    job.start();
  }

  fix() {
    const old = moment().subtract(2, "days");

    // console.log('You will see this message every second', Jobs);
    this.Jobs.find(
      {
        $and: [
          {
            updatedAt: {
              $gt: old.toDate()
            },
            "submission.state": "completed",
            type: this.re
          }
        ]
      },
      (err, jobs) => {
        jobs.forEach(job => {
          if (!job.results || !Object.keys(job.results).length) {
            job.readFromOutput(["statistics", "json"], (err, data) => {
              if (err) {
                log.error(job._id, err);
              } else {
                // TODO: could check to see if will save in mongo; if not, flag it
                // so that when need model re-reads the directory for results
                const json = JSON.parse(data);
                job.results = json || null;

                console.info("NEW RESULTS", json, job.results);

                job.save((err, savedJob) => {
                  if (err) {
                    log.error(err);
                    // return cb(saveErr, job);
                  }

                  //   return cb(null, savedJob);
                });
              }
            });
          }

          //   job.deleteOldIndex();
        });
      }
    );
  }

  _stop() {
    // TODO: anything?
  }
}

module.exports = JobFixer;
