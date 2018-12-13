const CronJob = require('cron').CronJob;
const moment = require('moment');

class JobCleaner {
    /* @param {Obj} config Job config
     * @param {Obj} comm Job Comm instance
     * @param {Obj} Model Mongoose model
     */
    constructor(Jobs) {
        this.Jobs = Jobs;

        this.re = new RegExp('annotation|saveFromQuery')
    }

    run() {
        // run every day at 8am 
        var job = new CronJob('0 0 8 * * *', this.clean.bind(this), null, true, 'America/New_York');
        job.start();
    };

    clean() {
        const old = moment().subtract(2, 'months');

        // console.log('You will see this message every second', Jobs);
        this.Jobs.find({

            $and: [{
                    updatedAt: {
                        $lt: old.toDate(),
                    }
                },
                {
                    type: this.re,
                }

            ],

            $or: [{
                    visibility: 'private'
                },
                {
                    visibility: {
                        $exists: false
                    }
                }
            ],

        }, '_id email search visibility name userID type', (err, jobs) => {
            jobs.forEach((job) => {
                console.info("archiving job: ", job);
                job.deleteOldIndex();
            });
        });
    };

    _stop() {
        // TODO: anything?
    }
}

module.exports = JobCleaner