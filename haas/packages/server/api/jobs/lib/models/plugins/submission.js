const _ = require('lodash');
const PromiseBird = require('bluebird');
const fs = PromiseBird.promisifyAll(require('fs-extra'));
const path = require('path');
const Schema = require.main.require('./common/database').Schema;

const log = require.main.require('./common/logger');

const tarballExtract = require.main.require('./api/jobs/lib/findInTarball');

const annotationQueue = require.main.require('./api/jobs/lib/annotationQueue');
const saveFromQueryQueue = require.main.require(
  './api/jobs/lib/saveFromQueryQueue'
);

const beanstalkWorker = require.main.require('./api/jobs/lib/beanstalkWorker');

const sqMailer = require.main.require('./common/mail');

const serverAddress = process.env.SERVER_ADDRESS;
const appName = process.env.APP_NAME;

const queueSchema = require.main.require(
  './api/jobs/lib/models/schemas/queueSchema'
);
const queueStates = queueSchema.states;

if (!(serverAddress && appName)) {
  throw new Error(
    `Require process.env to have SERVER_ADDRESS and APP_NAME prop/values`
  );
}

const lowerCaseAppName = appName.toLowerCase();

const events = {
  submitted: 'annotationSubmitted',
  started: 'annotationStarted',
  progress: 'annotationProgress',
  completed: 'annotationCompleted',
  failed: 'annotationFailed'
};

const queueType = 'annotation';
const annotationType = 'annotation';
// annotations can be forked from other annotations, on query
const saveType = 'saveFromQuery';

// TODO: not really sure what to do with this; maybe we should remove
// validations from the queueSchema?
// Some value to indicate that we're looking at an old or broken schema
const missingQueueID = '-9';

module.exports.events = events;

//TODO: improve logging to web interface. Make all errors log stack trace
module.exports.set = function submissionPlugin(schema, args) {
  const modelName = args.modelName;
  const jobComm = args.jobComm;

  schema.statics.listenAnnotationEvents = function listenAnnotationEvents() {
    _listenEvents.call(this);
  };

  schema.methods.submitAnnotation = function(cb) {
    const callback = cb || _.noop;

    // To clarify that "this" refers to the job document (Model instance)
    const jobDoc = this;

    // The annotation worker needs to know where to store our files
    // But this provides a less verbose way
    // I also like defining a smaller interface than the full schema
    // Makes behavior more predictable in my eyes
    const jobToSubmit = Object.assign({
      outputBasePath: path.join(jobDoc.dirs.out, jobDoc.name),
      options: jobDoc.options,
      assembly: jobDoc.assembly
    });

    let queue;

    if (jobDoc.inputQueryConfig && jobDoc.inputQueryConfig.queryBody) {
      Object.assign(jobToSubmit, jobDoc.inputQueryConfig.toObject());
      // toObject needed because in mongoose nested objects are mongoose schemas
      if (
        !jobToSubmit.fieldNames &&
        jobToSubmit.inputQuery &&
        jobToSubmit.indexName &&
        jobToSubmit.indexType
      ) {
        const err = new Error(
          'fieldNames, inputQuery, indexName, indexType required to submit job from a query'
        );
        log.error(err);
        return cb(err);
      }

      // Mongo doesn't like periods in field names, so we serialize the queryBody,
      // but workers expect no inner json
      jobToSubmit.queryBody = JSON.parse(jobToSubmit.queryBody);

      if (jobToSubmit.pipeline) {
        jobToSubmit.pipeline = JSON.parse(jobToSubmit.pipeline);
      }

      queue = saveFromQueryQueue;

      jobDoc.type = saveType;
    } else {
      jobToSubmit.inputFilePath = path.join(jobDoc.dirs.in, jobDoc.name);

      queue = annotationQueue;

      jobDoc.type = annotationType;
    }

    if (jobDoc.submission) {
      // TODO: archive the last submisison.
    }

    jobDoc.set('submission', queueSchema.instanceJobQueue());
    jobDoc.submission.type = queueType;

    jobDoc.submission.submittedDate = Date.now();
    console.info('about to save');
    jobDoc.save((saveErr, savedJob) => {
      if (saveErr) {
        console.info('save error', saveErr);
        log.error(saveErr);

        return jobDoc.deleteAnnotation(deleteErr => {
          if (deleteErr) {
            return callback(deleteErr);
          }

          return callback(saveErr);
        });
      }

      jobToSubmit.submissionID = savedJob.submission._id;
      console.info('submitting to queue');
      queue.submitClient.put(
        queue.priority,
        queue.delay,
        queue.timeToRun,
        JSON.stringify(jobToSubmit),
        (err, queueJobID) => {
          if (err) {
            log.error(err);
            // Using .call because this did not come from _listenEvents
            _onFailed.call(
              this,
              {
                _id: savedJob._id,
                reason: err.message
              },
              callback
            );

            return;
          }

          // @required props queueID, type
          // The user may have given us an empty queueSchema,
          // and maybe the queueSchema was updated to remove required = true
          // on type and queueID
          // We must set the properties directly on jobDoc.submission it seems;
          // tried doing this in the instanceJobQueue() function, didn't work
          savedJob.submission.queueID = queueJobID;

          savedJob.submission.log.messages.push('Job Submitted!');

          savedJob.submission.state = queueStates.submitted;

          savedJob.save(iSaveErr => {
            if (iSaveErr) {
              log.error(iSaveErr);

              savedJob.deleteAnnotation(deleteErr => {
                if (deleteErr) {
                  callback(deleteErr);
                  return;
                }

                callback(iSaveErr);
              });

              return;
            }

            // No email for submission
            jobComm.notifyUser(events.submited, savedJob, savedJob.submission);
            callback(null, savedJob);
          });
        }
      );
    });
  };

  // TODO: make API match, log error by itself
  // TODO: This also fails if the dirs are present, but not found
  schema.methods.deleteAnnotation = function(cb) {
    const dirs = this.dirs;

    try {
      if (this.dirs.out && fs.existsSync(this.dirs.out)) {
        fs.removeSync(this.dirs.out);
      }

      if (this.dirs.in && fs.existsSync(this.dirs.in)) {
        fs.removeSync(this.dirs.in);
      }
    } catch (err) {
      log.error(err);
      return cb(err, this);
    }

    this.remove((err, doc) => {
      if (err) {
        log.error(err);
        return cb(err);
      }

      return null, doc;
    });
  };

  schema.methods.readFromOutput = function(childPath, cb) {
    const outDir = this.dirs.out;
    const outFiles = this.outputFileNames;

    if (!outDir) {
      log.error('No output dir found', this);
      return cb(new Error('No output dir found'));
    }

    let file = Object.assign({}, outFiles);

    childPath.forEach(key => {
      file = file[key];
    });

    fs.readFile(path.join(outDir, file), 'utf8', cb);
  };

  schema.methods.checkAnnotationStatus = function checkAnnotationStatus(cb) {
    let job = this;

    annotationQueue.submitClient.stats_job(
      job.submission.queueID,
      (err, response) => {
        let changed = false;

        if (err === 'NOT_FOUND') {
          // This should never happen... except during schema transition
          // It's a malformed job; I'm not certain what the best course of action here is
          if (!job.submission.queueID) {
            log.error(new Error('No queueID found for job', job));
            job.submission.queueID = missingQueueID;

            changed = true;
          }

          // This should never happen... except during schema transition
          if (!job.submission.type) {
            log.error(new Error('No type found for job', job));
            job.submission.type = queueType;

            changed = true;
          }

          // When database is not in good state, jobs may go to GONE improperly
          if (
            job.submission.state !== queueStates.completed &&
            job.submission.state !== queueStates.failed
          ) {
            job.submission.state = queueStates.gone;
            changed = true;
          }
        } else {
          log.debug('received in peek', err, response);

          if (response.reserves != job.submission.attempts) {
            job.submission.attempts = response.reserves;
            changed = true;
          }

          if (
            response.state === 'reserved' &&
            job.submission.state !== queueStates.started
          ) {
            job.submission.state = queueStates.started;
            changed = true;

            _email(job);
          } else if (
            response.state === 'ready' &&
            job.submission.state !== queueStates.submitted
          ) {
            job.submission.state = queueStates.submitted;
            changed = true;
          } else if (
            response.state === 'buried' &&
            job.submission.state !== queueStates.failed
          ) {
            job.submission.state = queueStates.failed;
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
          return cb(null, job);
        }
      }
    );
  };

  schema.statics.completeJob = function completeJob(message, cb) {
    if (!message.results) {
      log.error(`No results found in schema.statics.completeJob`);
    }

    if (!message.submissionID) {
      log.error(
        `completion message received without submissionID, can't id job`,
        message
      );
      return cb(`completion message received without submissionID`);
    }

    this.findOneAndUpdate(
      { 'submission._id': message.submissionID },
      {
        'submission.state': queueStates.completed,
        'submission.finishedDate': Date.now(),
        outputFileNames: message.results.outputFileNames
      },
      { new: true },
      (saveErr, job) => {
        if (saveErr) {
          log.error(saveErr);
          return cb(saveErr, job);
        }

        // Delete the input file to save space
        // These files are never re-used
        // If this fails, it's not important to notify the user
        // but rather important to notify the admin
        if (job.dirs.in) {
          fs.emptyDir(job.dirs.in, err => {
            if (err) {
              log.error(err);
            }
          });
        }

        job.readFromOutput(['statistics', 'json'], (err, data) => {
          if (err) {
            log.error(err);
            job.submission.log.messages.push("Couldn't read statistics file");
            job.results = {};
          } else if (!data) {
            job.submission.log.messages.push('No statistics found');
            job.results = {};
          } else {
            // TODO: could check to see if will save in mongo; if not, flag it
            // so that when need model re-reads the directory for results
            const json = JSON.parse(data);
            job.results = json || null;
          }

          job.save((err, savedJob) => {
            if (err) {
              log.error(err);
              return cb(saveErr, job);
            }

            return cb(null, savedJob);
          });
        });
      }
    );
  };

  // TODO: These don't all need to be public, but do need access to findOne
  // which is only availble on the Model, which is the execution context (this)
  // of schema.statics
  schema.statics.startJob = function recordStartJob(message, cb) {
    if (!message.submissionID) {
      log.error(
        `startJob message received without submissionID, can't id job`,
        message
      );
      return cb(`startJob message received without submissionID`);
    }

    this.findOneAndUpdate(
      { 'submission._id': message.submissionID },
      {
        'submission.state': queueStates.started,
        'submission.startedDate': Date.now(),
        $push: {
          'submission.log.messages':
            'An Amazon cloud worker has picked up your annotation.'
        },
        $inc: { 'submission.attempts': 1 },
        config: JSON.stringify(message.jobConfig)
      },
      { new: true },
      (saveErr, job) => {
        if (saveErr) {
          log.error(saveErr);
          return cb(saveErr);
        }

        return cb(null, job);
      }
    );
  };

  // TODO: Figure out how to use findOneAndUpdate, becuase we need to $set submission
  // only if it doesn't exist, which can happen if the submission failed, and that's
  // what triggered failJob
  schema.statics.failJob = function failJob(message, cb) {
    const callback = cb || _.noop;

    // We accept queueID or _id because job can fail before submitted to the queue
    if (!message.submissionID && !message.queueID) {
      log.error(
        `index fail message received without submissionID or queueID`,
        message
      );
      return cb(`index fail message received without submissionID or queueID`);
    }

    let find;
    if (!message.queueID) {
      find = { 'submission.queueID': message.queueID };
    } else {
      find = { 'submission._id': message.submissionID };
    }

    this.findOne(find, function getJobCb(err, job) {
      if (err) {
        log.error(err);
        return cb(err, null);
      }

      // From the perspective of this function, not finding a job isn't an error
      // It's just missing data
      if (!job) {
        log.warn(
          `No job found with submission.submissionID == ${message.submissionID}`
        );
        return callback(null, null);
      }

      // If we fail because of a submission schema validation error,
      // This will not be populated yet.
      if (!job.submission) {
        job.set('submission', queueSchema.instanceJobQueue());
      }
      /* Job states are idempotent */
      if (job.submission.state === queueStates.failed) {
        log.debug(`job ${job._id} received duplicate failed message`);
        return cb(null, job, 1);
      }

      job.submission.finishedDate = Date.now();

      if (message.reason) {
        //https://davidwalsh.name/combining-js-arrays
        //cant use concat without clone, triggers TypeError issue
        job.set('submission.log.exception', message.reason);
      } else {
        log.warn('failJob expects jobObj to have log.exceptions');
      }

      job.submission.state = queueStates.failed;

      // Delete the input file to save space
      // These files are never re-used
      // If this fails, it's not important to notify the user
      // but rather important to notify the admin
      if (job.dirs.in) {
        fs.emptyDir(job.dirs.in, err => {
          if (err) {
            log.error(err);
          }
        });
      }

      job.save(saveErr => {
        if (saveErr) {
          log.error(saveErr);
          return cb(saveErr, null);
        }

        return cb(null, job);
      });
    });
  };

  function _onStarted(message, cb) {
    this.startJob(message, function recordStartJobdCb(err, job, duplicate) {
      //TODO: can't notify the user here, since jobID doesn't contain enough info
      if (err) {
        log.error(err);
        return cb(err);
      }

      // Job may not have been found; there's not much to do about this except log
      // Which happens in startJob
      if (!job) {
        return cb(null, null);
      }

      if (!duplicate) {
        _notifyUserWithEmail(events.started, job, job);
      }

      cb(null, job);
    });
  }

  function _onCompleted(message, cb) {
    this.completeJob(message, function completedCb(err, job, duplicate) {
      //we can notify the user here, ince jobObj contains enough info
      if (err) {
        log.error(err);
        //TODO notifyFailed(jobObjOrID, err.message); if makes sense...
        return cb(err, null);
      }

      // Job may not have been found; there's not much to do about this except log
      // Which happens in completeJob
      if (!job) {
        return cb(null, null);
      }

      if (!duplicate) {
        _notifyUserWithEmail(events.completed, job, job);

        // Tell listeners that htis job is finished;
        schema.emit(events.completed, job);
      }

      return cb(null, job);
    });
  }

  function _onFailed(message, cb) {
    const callback = cb || _.noop;

    this.failJob(message, function failedCb(err, job, duplicate) {
      //we can notify the user here, ince jobObj contains enough info
      if (err) {
        //TODO: send to user; will need to get the full job obj from mongo first
        //jobObjOrID, but does that make sense given err only when failJob fails to work?
        // if (!jobObj.log && jobObj.log.exceptions) {
        //   //not logging because why would the user care?
        //   log.error(new Error('redis job record didn\'t have log.exceptions') );
        // }

        if (job) {
          console.info('failed job', job);
          job.submission.log.exception = err.message;
          _notifyUserWithEmail(events.failed, job, job);
        }

        return callback(err, null);
      }

      // Can't do much; we found no job
      // Inform the beanstalk queue server however that nothing bad happened
      // Explictly returning "null" for job to indicate intent
      if (!job) {
        return callback(null, null);
      }

      if (!duplicate) {
        _notifyUserWithEmail(events.failed, job, job);
      }

      return callback(null, job);
    });
  }

  //unlike the other events, we don't couple saving to emitting
  //this is to avoid performance issues, since updates may be very frequent
  //expects either a string, which gets stored as a message
  //or an object, in which case it stores key => value
  //we expect jobComm to handle this messaging

  //TOOD: think about making a more robust version
  function _onProgress(message, cb) {
    if (!message.submissionID) {
      log.error(
        `_onProgress message received without submissionID, can't id job`,
        message
      );
      return cb(`_onProgress message received without submissionID`);
    }

    if (!message.data) {
      log.error(`_onProgress message received without data`);
      return cb(`_onProgress message received without data`);
    }

    let progressUpdate = {
      'submission.state': queueStates.started
    };

    if (typeof message.data === 'string') {
      progressUpdate = Object.assign(progressUpdate, {
        $push: { 'submission.log.messages': message.data }
      });
    } else if (typeof message.data === 'object') {
      //TODO: Expand to support arbitrary property
      if (message.data.hasOwnProperty('progress')) {
        progressUpdate['submission.log.progress'] = message.data.progress;
      }

      if (message.data.hasOwnProperty('skipped')) {
        progressUpdate['submission.log.skipped'] = message.data.skipped;
      }
    }

    this.findOneAndUpdate(
      { 'submission._id': message.submissionID },
      progressUpdate,
      {
        new: true
      },
      (saveErr, job) => {
        // Todo: implement retry mechanism
        if (!job) {
          const err = new Error("Couldn't find job for submissionID", message);
          log.error(err);
          return cb(err);
        }

        if (saveErr) {
          log.error(saveErr);
          job.submission.log.exception = 'Failed to record progress';
          jobComm.notifyUser(events.progress, job, job.submission);
          return cb(saveErr);
        }

        //Could also just send back message.data, but I like the idea
        // of real state updates, in case the interface is out of sync
        jobComm.notifyUser(events.progress, job, job.submission);

        return cb(null, job);
      }
    );
  }

  function _notifyUserWithEmail(event, jobDoc, data) {
    jobComm.notifyUser(event, jobDoc, data);
    _email(jobDoc);
  }

  function _listenEvents() {
    const startedFunc = _onStarted.bind(this);
    const completedFunc = _onCompleted.bind(this);
    const failedFunc = _onFailed.bind(this);
    const progressFunc = _onProgress.bind(this);

    [annotationQueue, saveFromQueryQueue].forEach(queue => {
      beanstalkWorker.start(
        queue.eventsClient,
        queue.maxReserves,
        (err, message, cb) => {
          if (err) {
            // If we don't have a message, not much to do
            log.error(err);
            return;
          }

          if (!message) {
            log.error('no message found');
            return;
          }

          if (!message.event) {
            log.error(`payload required, none found for job id ${message.id}`);
            return cb(`payload required, none found for job id ${message.id}`);
          }

          if (message.event === annotationQueue.events.started) {
            return startedFunc(message, cb);
          }

          if (message.event === annotationQueue.events.completed) {
            return completedFunc(message, cb);
          }

          if (message.event === annotationQueue.events.failed) {
            return failedFunc(message, cb);
          }

          if (message.event === annotationQueue.events.progress) {
            return progressFunc(message, cb);
          }
        }
      );
    });
  }

  function _email(jobObj) {
    if (!jobObj.email) {
      return;
    }

    let address;
    let message;
    const subject = `${jobObj.name} annotation update`;

    if (jobObj.submission.state === queueStates.completed) {
      address = `<a href="${serverAddress}/results?_id=${
        jobObj._id
      }">${lowerCaseAppName}</a>`;

      const stuff = `This job, created on ${jobObj._id.getTimestamp()} is complete!`;
      message = `${stuff}.\nVisit ${address} to view or download your results`;
    } else if (jobObj.submission.state === queueStates.submitted) {
      let stuff;

      if (jobObj.submission.attempts > 1) {
        stuff = `This job, created at ${jobObj._id.getTimestamp()} has been re-submitted`;
      } else {
        stuff = `Good news! This job, created at ${jobObj._id.getTimestamp()} has been submitted.`;
      }

      address = `<a href="${serverAddress}/queue?_id=${
        jobObj._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nVisit ${address} to track progress`;
    } else if (jobObj.submission.state === queueStates.started) {
      let stuff;

      if (jobObj.submission.attempts > 1) {
        stuff = `This job, created at ${jobObj._id.getTimestamp()} has been re-started`;
      } else {
        stuff = `Good news! This job, created on ${jobObj._id.getTimestamp()} has been started!`;
      }

      address = `<a href="${serverAddress}/queue?_id=${
        jobObj._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nVisit ${address} to track progress`;
    } else if (jobObj.submission.state === queueStates.failed) {
      const stuff = `I'm sorry! This job, created on ${jobObj._id.getTimestamp()} has failed`;

      address = `<a href="${serverAddress}/failed?_id=${
        jobObj._id
      }">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nIt failed because of ${
        jobObj.submission.log.exception
      }.\n
      Visit ${address} to see the full job log`;
    } else if (jobObj.submission.state === queueStates.gone) {
      const stuff = `I'm sorry! This job, created on ${jobObj._id.getTimestamp()} has gone missing`;

      address = `<a href="${serverAddress}/submit">${lowerCaseAppName}</a>`;

      message = `${stuff}.\nWe're not sure what happend. Please visit ${address} to try again`;
    }

    sqMailer.send(jobObj.email, subject, message, appName, err => {
      if (err) {
        log.error(`Failed to send job status update email because ${err}`);
      }
    });
  }
};

// //TODO: improve logging to web interface. Make all errors log stack trace
