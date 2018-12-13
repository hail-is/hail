const _ = require("lodash");
const fs = require("fs");
var mongoose = require("mongoose");
var Schema = mongoose.Schema;

const log = require.main.require("./common/logger");

const indexQueue = require.main.require("./api/jobs/lib/indexQueue");

const queueSchema = require.main.require(
  "./api/jobs/lib/models/schemas/queueSchema"
);
const queueStates = queueSchema.states;

const beanstalkWorker = require.main.require("./api/jobs/lib/beanstalkWorker");

const sqMailer = require.main.require("./common/mail");

const serverAddress = process.env.SERVER_ADDRESS;
const appName = process.env.APP_NAME;

if (!(serverAddress && appName)) {
  throw new Error(
    `Require process.env to have SERVER_ADDRESS and APP_NAME prop/values`
  );
}

const lowerCaseAppName = appName.toLowerCase();

// TODO: Improve idempotency. If we get a started event after a completed event
// Need to make sure we don't override

const events = {
  submitted: "indexSubmitted",
  started: "indexStarted",
  progress: "indexProgress",
  completed: "indexCompleted",
  failed: "indexFailed"
};

const queueType = "searchIndex";

const cDate = new Date();
const b11 = new Date(2018, 7, 6);

const version = cDate.getTime() >= b11.getTime() ? 11 : 10;

//TODO: improve logging to web interface. Make all errors log stack trace
module.exports.set = function indexCommPlugin(schema, args) {
  const jobComm = args.jobComm;
  const annotationEvents = args.jobSubmissionEvents;
  const modelName = args.modelName; //mongoose.model(args.modelName, schema);

  schema.statics.listenIndexEvents = function listen() {
    _listenEvents.call(this);
  };

  schema.statics.checkIndexStatus = function checkIndexStatus(queryObject, cb) {
    return this.findOne(queryObject, (err, job) => {
      if (err) {
        return cb(err, null);
      }

      if (!job) {
        const err = new Error(new Error(`No job found with _id: ${job._id}`));
        log.error(err);

        return cb(err, null);
      }

      if (!job.search) {
        const err = new Error(
          `No job search object found for job with _id: ${job._id}`
        );
        log.error(err);

        return cb(err, null);
      }

      // This is more of a warning: the api could change; the active submission
      // may be removed after finished
      if (!job.search.activeSubmission) {
        // if it's simply archived, no issue
        if (job.type !== queueStates.archived) {
          return cb(null, null);
        }

        const err = new Error(
          `No job search activeSubmission object found for job with _id: ${
            job._id
          }`
        );
        log.warn(err);

        return cb(null, job);
      }

      const searchIndex = job.search.activeSubmission;

      indexQueue.submitClient.stats_job(
        job.search.activeSubmission.queueID,
        (err, response) => {
          let changed = false;
          if (err === "NOT_FOUND") {
            if (
              searchIndex.state !== queueStates.completed &&
              searchIndex.state !== queueStates.failed
            ) {
              searchIndex.state = queueStates.gone;
              changed = true;

              _email(job);
            }
          } else {
            log.debug("received in peek", err, response);

            if (response.reserves != searchIndex.attempts) {
              searchIndex.attempts = response.reserves;
              changed = true;
            }

            if (
              response.state === "reserved" &&
              searchIndex.state !== queueStates.started
            ) {
              searchIndex.state = queueStates.started;
              changed = true;

              _email(job);
            } else if (
              response.state === "ready" &&
              searchIndex.state !== queueStates.submitted
            ) {
              searchIndex.state = queueStates.submitted;
              changed = true;

              _email(job);
            } else if (
              response.state === "buried" &&
              searchIndex.state !== queueStates.failed
            ) {
              searchIndex.state = queueStates.failed;
              changed = true;

              _email(job);
            }
          }

          if (changed) {
            job.save(saveErr => {
              if (saveErr) {
                log.error(saveErr);
                return cb(saveErr);
              }

              return cb(null, job);
            });
          } else {
            return cb(err, job);
          }
        }
      );
    });
  };

  schema.methods.setOldIndexToArchive = function setIndexArchive() {
    const jobDoc = this;

    if (jobDoc.search && jobDoc.search.activeSubmission) {
      // copy object, in order to guard against modification by reference below
      const oldSubmission = Object.assign({}, jobDoc.search.activeSubmission);

      // TODO: not sure why this can happen; investigate
      if (!oldSubmission.type) {
        oldSubmission.type = queueType;
      }

      if (!jobDoc.search.archivedSubmissions) {
        jobDoc.set({
          "search.archivedSubmissions": [oldSubmission]
        });
      } else {
        jobDoc.search.archivedSubmissions.push(oldSubmission);
      }

      jobDoc.search.activeSubmission = undefined;
    }

    return jobDoc;
  };

  schema.statics.submitIndexJob = function submitIndexJob(
    jobDoc,
    addedFiles,
    cb
  ) {
    const callback = cb || _.noop;

    if (
      jobDoc.search &&
      jobDoc.search.activeSubmission &&
      (jobDoc.search.activeSubmission.state === queueStates.submitted ||
        jobDoc.search.activeSubmission.state === queueStates.started)
    ) {
      const err = new Error(
        "Cannot submit new index job until previous one finishes"
      );
      log.error(err);

      callback(err);
      return;
    }

    jobDoc.deleteOldIndex(err => {
      if (err) {
        log.error(err);
        callback(err, null);
        return;
      }

      const indexName = `${jobDoc._id}_${jobDoc.userID}`;
      const indexType = jobDoc._id;

      if (!jobDoc.search) {
        jobDoc.set("search", {
          indexName,
          indexType
        });
      }

      if (!jobDoc.search.indexName) {
        jobDoc.search.indexName = indexName;
      }

      let existingIndexConfig;
      if (jobDoc.type === "saveFromQuery") {
        if (version >= 11 && !jobDoc.inputQueryConfig.indexConfig) {
          const err = new Error(
            "If job has inputQueryConfig, must also have indexConfig"
          );
          log.error(err);
          callback(err, null);
          return;
        }

        existingIndexConfig = jobDoc.inputQueryConfig.indexConfig;
      }

      if (!jobDoc.search.indexType) {
        // The type is the job id
        jobDoc.search.indexType = indexType;
      }

      // We always submit a new activeSubmission; above we previously archived an
      // existing submission
      jobDoc.set("search.activeSubmission", queueSchema.instanceJobQueue());

      jobDoc.search.activeSubmission.type = queueType;

      jobDoc.search.activeSubmission.submittedDate = Date.now();

      jobDoc.save((saveErr, savedJob) => {
        if (saveErr) {
          log.error(saveErr);

          this.failIndexJob(
            {
              _id: jobDoc._id,
              reason: saveErr.message
            },
            callback
          );

          return;
        }

        // What we submit to the queue
        const jobToSubmit = {
          inputDir: savedJob.dirs.out,
          inputFileNames: savedJob.outputFileNames,
          indexName: savedJob.search.indexName,
          indexType: savedJob.search.indexType,
          submissionID: savedJob.search.activeSubmission._id,
          assembly: savedJob.assembly
        };

        if (existingIndexConfig) {
          jobToSubmit.indexConfig = existingIndexConfig;
        }

        for (let key in jobToSubmit) {
          if (
            !jobToSubmit[key] ||
            (Array.isArray(jobToSubmit[key]) && !jobToSubmit[key].length)
          ) {
            const err = new Error(`Cannot index without: ${key} value`);
            log.error(err);

            this.failIndexJob(
              {
                _id: savedJob._id,
                reason: err.message
              },
              callback
            );
            return;
          }
        }

        // If this is a Nth indexing job, we may have an existing fieldNames list
        // to which the user presumably wants to add some data
        // So we will pass these to the indexing job worker, which will append
        // any new fields
        if (savedJob.search.fieldNames && savedJob.search.fieldNames.length) {
          jobToSubmit.fieldNames = savedJob.search.fieldNames;
        }

        if (addedFiles) {
          let err;

          // let addTheseFiles = Array.isArray(addedFiles)
          //   ? addedFiles
          //   : [addedFiles];

          for (let i = 0; i < addedFiles.length; i++) {
            if (!_.isObject(addedFiles[i])) {
              err = new Error("Added files must be array");
              break;
            }
          }

          // Must use call, because used outside listenEvents
          if (err) {
            log.error(err);

            this.failIndexJob(
              {
                _id: savedJob._id,
                reason: err.message
              },
              callback
            );
            return;
          }

          jobToSubmit.addedFiles = addedFiles;
        }

        indexQueue.submitClient.put(
          indexQueue.priority,
          indexQueue.delay,
          indexQueue.timeToRun,
          JSON.stringify(jobToSubmit),
          (err, queueJobId) => {
            // this is the instance of the model here

            // Notice that we use jobDoc below, to make sure that we don't delete
            // the results, which we stripped above
            if (err) {
              log.error(err);

              // Saves the job as failed
              this.failIndexJob(
                {
                  _id: savedJob._id,
                  reason: err.message
                },
                callback
              );
              return;
            }

            savedJob.search.activeSubmission.queueID = queueJobId;

            savedJob.search.activeSubmission.state = queueStates.submitted;

            savedJob.search.activeSubmission.log.messages.push(
              "Index Job Submitted!"
            );

            savedJob.save((innerErr, job) => {
              if (innerErr) {
                log.error(innerErr);

                this.failIndexJob(
                  {
                    _id: savedJob._id,
                    reason: saveErr.message
                  },
                  callback
                );

                return;
              }

              jobComm.notifyUser(events.submitted, job, job.search);
            });
          }
        );
      });
    });
  };

  schema.statics.completeIndexJob = function completeJob(
    message,
    cb,
    attempts = 0
  ) {
    const callback = cb || _.noop;

    if (!message.results) {
      log.error(`No results found in schema.statics.completeJob`);
    }

    if (!message.submissionID) {
      log.error(
        `index completion message received without queueID, can't id job`,
        message
      );
      return cb(`index completion message received without queueID`);
    }

    this.findOneAndUpdate(
      {
        "search.activeSubmission._id": message.submissionID
      },
      {
        "search.fieldNames": message.fieldNames,
        "search.activeSubmission.finishedDate": Date.now(),
        "search.activeSubmission.state": queueStates.completed,
        $push: {
          "search.indexConfig": JSON.stringify(message.indexConfig)
        }
      },
      {
        new: true
      },
      (getErr, job) => {
        if (getErr) {
          if (attempts < 10) {
            setTimeout(() => {
              schema.statics.completeIndexJob(message, cb, ++attempts);
            }, 20);
            return;
          }
          log.error(getErr);
          cb(getErr);
          return;
        }

        if (!job) {
          const err = new Error("Couldn't find job");
          log.error("couldn't find job in completeIndexJob");
          cb(err, null);
          return;
        }

        jobComm.notifyUser(events.completed, job, job.search);
        cb(null, job);
      }
    );
  };

  schema.statics.startIndexJob = function startIndexJob(
    message,
    cb,
    attempts = 0
  ) {
    const callback = cb || _.noop;

    if (!message.submissionID) {
      log.error(
        `index start message received without submissionID, can't id job`,
        message
      );
      return cb(`index start message received without submissionID`);
    }

    //TODO: figure out how to do atomic check of state, if say updates sent out of order
    this.findOneAndUpdate(
      {
        "search.activeSubmission._id": message.submissionID
      },
      {
        "search.activeSubmission.state": queueStates.started,
        "search.activeSubmission.startedDate": Date.now(),
        $push: {
          "search.activeSubmission.log.messages": "Job Started!"
        },
        $inc: {
          "search.activeSubmission.attempts": 1
        }
      },
      {
        new: true
      },
      (saveErr, job) => {
        // TODO: implement retry logic
        if (saveErr) {
          if (attempts < 3) {
            setTimeout(() => {
              schema.statics.startIndexJob(message, cb, ++attempts);
            }, 200);
          } else {
            log.error("findOneAndUpdate error:", saveErr);

            return this.failIndexJob(
              {
                _id: job._id,
                reason: `Couldn't start indexing request, because: ${
                  saveErr.message
                }`
              },
              callback
            );
          }
        } else if (!job) {
          const err = new Error(
            "Couldn't locate job for message with id " + message &&
              message.submissionID
          );
          log.error(err);

          return cb(err);
        } else {
          jobComm.notifyUser(events.started, job, job.search);
          return cb(null, job);
        }
      }
    );
  };

  schema.statics.failIndexJob = function failJob(message, cb, attempts = 0) {
    const callback = cb || _.noop;

    // We accept queueID or _id because job can fail before submitted to the queue
    if (!(message.submissionID || message._id)) {
      const err = `failIndexJob message received without submissionID or _id`;
      log.error(err, message);

      return callback(err, null);
    }

    let find = {};

    if (message._id) {
      find._id = message._id;
    } else {
      find = {
        "search.activeSubmission._id": message.submissionID
      };
    }

    this.findOneAndUpdate(
      find,
      {
        "search.activeSubmission.state": queueStates.failed,
        "search.activeSubmission.finishedDate": Date.now(),
        "search.activeSubmission.log.exception": message.reason
      },
      {
        new: true
      },
      (saveErr, job) => {
        // TODO: add retry logic
        if (saveErr) {
          if (attempts < 5) {
            setTimeout(() => {
              schema.statics.failIndexJob(message, cb, ++attempts);
            }, 200);
          }
        } else if (!job) {
          const err = new Error(
            "Couldn't locate job for message with id " + message &&
              message.submissionID
          );

          log.error(err);
          return cb(err);
        } else {
          _notifyUserWithEmail(events.failed, job, job.search);
          return cb(null, job);
        }
      }
    );
  };

  //unlike the other events, we don't couple saving to emitting
  //this is to avoid performance issues, since updates may be very frequent
  //expects either a string, which gets stored as a message
  //or an object, in which case it stores key => value
  //we expect jobComm to handle this messaging

  //TOOD: think about making a more robust version
  function _onProgress(message, cb, attempts = 0) {
    if (!message.submissionID) {
      return cb(`index _onProgress message received without submissionID`);
    }

    if (!message.data) {
      const err = new Error(`_onProgress message received without data`);
      log.error(err);
      return cb(err);
    }

    let progressUpdate;
    if (typeof message.data === "string") {
      progressUpdate = {
        $push: {
          "search.activeSubmission.log.messages": message.data
        }
      };
    } else if (typeof message.data === "object") {
      // We don't currently support skipping
      progressUpdate = {
        "search.activeSubmission.log.progress": message.data.progress
      };
    }

    // I believe these are atomic, which should reduce possibility of version conflicts
    this.findOneAndUpdate(
      {
        "search.activeSubmission._id": message.submissionID
      },
      progressUpdate,
      {
        new: true
      },
      (saveErr, job) => {
        if (saveErr) {
          // can fail, concurrent operations
          // http://www.mattpalmerlee.com/2014/03/22/a-pattern-for-handling-concurrent/
          if (attempts < 3) {
            setTimeout(() => {
              _onProgress.call(this, message, cb, ++attempts);
            }, 200);
            return;
          }
          log.error(saveErr);

          cb(saveErr, null);
          return;
        }

        if (!job) {
          const id = (message && message.submissionID) || "unknown id";
          const err = new Error(
            `Couldn't locate job for message with id: ${id}`
          );

          log.error(err);

          cb(err, null);
          return;
        }

        jobComm.notifyUser(events.progress, job, job.search.activeSubmission);

        cb(null, job);
      }
    );
  }

  function _listenEvents() {
    const progressFunc = _onProgress.bind(this);

    schema.on(annotationEvents.completed, jobObj => {
      this.submitIndexJob(jobObj);
    });

    beanstalkWorker.start(
      indexQueue.eventsClient,
      indexQueue.maxReserves,
      (err, message, cb) => {
        if (err) {
          log.error(err);
          return;
        }

        if (!(message && message.event)) {
          log.error(`payload required, none found for job id`, message);
          return;
        }

        if (message.event === indexQueue.events.started) {
          return this.startIndexJob(message, cb);
        }

        if (message.event === indexQueue.events.completed) {
          return this.completeIndexJob(message, cb);
        }

        if (message.event === indexQueue.events.failed) {
          return this.failIndexJob(message, cb);
        }

        if (message.event === indexQueue.events.progress) {
          return progressFunc(message, cb);
        }
      }
    );
  }

  function _notifyUserWithEmail(event, jobDoc, data) {
    jobComm.notifyUser(event, jobDoc, data);
    _email(jobDoc);
  }

  function _email(jobDoc) {
    if (!jobDoc.email) {
      return;
    }

    const subject = `${jobDoc.name} annotation update`;

    let address;
    let message;

    const searchIndex = jobDoc.search.activeSubmission;

    if (searchIndex.state === queueStates.completed) {
      const stuff = `This job, created on ${jobDoc._id.getTimestamp()}, finished indexing!`;
      address = `<a href="${serverAddress}/results?_id=${
        jobDoc._id
      }&search=true">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nVisit ${address} to view or download your results`;
    } else if (searchIndex.state === queueStates.submitted) {
      let stuff;
      if (searchIndex.attempts > 1) {
        stuff = `This job, created at ${jobDoc._id.getTimestamp()} is being re-indexed`;
      } else {
        stuff = `Good news! This job, created at ${jobDoc._id.getTimestamp()} is indexing.`;
      }

      address = `<a href="${serverAddress}/queue?_id=${
        jobDoc._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nVisit ${address} to track progress`;
    } else if (searchIndex.state === queueStates.started) {
      let stuff;
      if (searchIndex.attempts > 1) {
        stuff = `This job, created at ${jobDoc._id.getTimestamp()} has started indexing`;
      } else {
        stuff = `Good news! Your job, created on ${jobDoc._id.getTimestamp()} is now being prepared for search!`;
      }

      address = `<a href="${serverAddress}/queue?_id=${
        jobDoc._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nVisit ${address} to track progress`;
    } else if (searchIndex.state === queueStates.failed) {
      const stuff = `I'm sorry! This job, created on ${jobDoc._id.getTimestamp()} has failed to index`;

      address = `<a href="${serverAddress}/failed?_id=${
        jobDoc._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nIt failed because of ${searchIndex.log.exception ||
        "an error"}.
       Visit ${address} to see the full job log`;
    } else if (searchIndex.state === queueStates.gone) {
      const stuff = `I'm sorry! This job, created on ${jobDoc._id.getTimestamp()} indexing request has gone missing`;

      address = `<a href="${serverAddress}/submit">${lowerCaseAppName}</a>`;

      message = `${stuff}.We're not sure what happend.`; /* Please visit ${address} to try again`;*/
    }

    sqMailer.send(jobDoc.email, subject, message, appName, err => {
      if (err) {
        log.error(`Failed to send job status update email because ${err}`);
      }
    });
  }
};
