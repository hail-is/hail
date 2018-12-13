/**
 * Using Rails-like standard naming convention for endpoints.
 * GET     /things              ->  index
 * POST    /things              ->  create
 * GET     /things/:id          ->  show
 * PUT     /things/:id          ->  update
 * DELETE  /things/:id          ->  destroy
 */
// TODO: replace multiple exports calls with a single one

// @public
// TODO: separate router calls from utility functions

// TODO: move away from use of public ID, it's wasteful
// Search by ID only, and then match the req.user.id to the job.userID
const log = require.main.require("./common/logger");
const config = require.main.require("./common/config").jobs;
const configPath = require.main.require("./common/config").jobs.configDir;

const yaml = require("js-yaml");
const path = require("path");
const fs = require("fs");

const queueStates = require.main.require(
  "./api/jobs/lib/models/schemas/queueSchema"
).states;

const incompleteStateRegex = new RegExp(
  `${queueStates.submitted}|${queueStates.started}`
);
const completeStateRegex = queueStates.completed;
const failedStateRegex = new RegExp(
  `${queueStates.failed}|${queueStates.gone}`
);

const { ObjectID } = require("mongodb");

const errCode = 500;

const elasticConfig = Object.assign({}, config.elastic);

const elasticsearch = require("elasticsearch");

const elasticClient = new elasticsearch.Client(elasticConfig);

// TODO: allow configuration of job expiration
const jobExpireSafety = 2 * 24 * 60 * 60 * 1000;
const jobExpireExtension = 5 * 24 * 60 * 60 * 1000;

const projection = {
  __v: 0,
  dirs: 0,
  "inputQueryConfig.indexName": 0,
  "inputQueryConfig.indexType": 0,
  "search.indexName": 0,
  "search.indexType": 0
};

// TODO: simplify this; just use a graphql-like aproach; pass down the full object you want?
// TODO: Vastly simplify permissions; should live in a single call to the model layer
exports = module.exports = function jobsCtrl(Jobs, jobComm) {
  return {
    getOne,
    getJobs,
    searchJob,
    deleteOne,
    reIndex,
    /*restart,*/
    update,
    checkAnnotationStatus,
    checkIndexStatus,
    getConfig,
    saveFromQuery,
    addSynonyms
  };

  function getJobs(req, res) {
    const query = {};
    let wantShared = false;
    if (req.params.type === "incomplete") {
      query["submission.state"] = incompleteStateRegex;
    } else if (
      req.params.type === "complete" ||
      req.params.type === "completed"
    ) {
      query["submission.state"] = completeStateRegex;
    } else if (req.params.type === "failed") {
      query["submission.state"] = failedStateRegex;
    } else if (req.params.type === "deleted") {
      query.type = "deleted";
    } else if (req.params.type === "shared") {
      wantShared = true;
    } else if (req.params.type !== "all") {
      return _noJob(res);
    }

    if (req.params.visibility) {
      if (req.params.visibility === "private") {
        query.visibility = "private";
      } else if (req.params.visibility === "public") {
        query.visibility = "public";
      } else if (req.params.visibility) {
        return _noJob(res);
      }
    }

    if (query.visibility !== "public") {
      if (req.user) {
        query.userID = req.user.id;
      } else {
        return _notAuthorized(res);
      }
    }

    if (wantShared) {
      delete query.userID;

      query[`sharedWith.${req.user.id}`] = {
        $gte: 400
      };
    }

    if (!query.type) {
      query.type = {
        $ne: "deleted"
      };
    }

    // Fast, but we currently need a better solution to populate the job name
    Jobs.find(query, {
      results: 0,
      dirs: 0,
      __v: 0,
      "inputQueryConfig.indexName": 0,
      "inputQueryConfig.indexType": 0,
      "search.indexName": 0,
      "search.indexType": 0
    })
      .populate("_creator", "_id name createdAt")
      .lean()
      .exec((err, jobs) => _tellUser(res, err, jobs));
  }

  function getConfig(req, res) {
    try {
      const doc = yaml.safeLoad(
        fs.readFileSync(path.join(configPath, `${req.params.assembly}.yml`))
      );

      _tellUser(res, null, doc.User);
    } catch (e) {
      log.error(e);
      _tellUser(res, e, null);
    }
  }

  function saveFromQuery(req, res) {
    if (!req.params.id) {
      _noJob(res);
      return;
    }

    //Must be logged in, even if the user does not own this job
    //Since will be creating a new job
    if (!req.user) {
      _notAuthorized(res);
      return;
    }

    // 10 minutes timeout just for POST to myroute
    req.socket.setTimeout(10 * 60 * 1000);

    // upload request the request, {Bool} saveJob : optonsla, {fn} cb : optional
    Jobs.findById(req.params.id, (err, job) => {
      if (err || !job) {
        _tellUser(res, err, job);
        return;
      }

      if (!job.isPermittedFor(req.user && req.user.id)) {
        _notAuthorized(res);
        return;
      }

      if (
        !job.search &&
        job.search.indexName &&
        job.search.indexType &&
        job.search.fieldNames &&
        job.search.fieldNames.length
      ) {
        res.status(404).end();
        return;
      }

      const name = req.body.name || `${job.name}-savedFromQuery`;

      const { email } = req.user;

      const jobToSubmit = {
        _creator: job._id,
        assembly: job.assembly,
        name,
        email,
        inputQueryConfig: {
          queryBody: JSON.stringify(req.body.inputQueryBody),
          indexName: job.search.indexName,
          indexType: job.search.indexType,
          // the latest configuration
          indexConfig: job.search.indexConfig[0],
          fieldNames: job.search.fieldNames
        },
        outputBaseFileName: req.body.outputBaseFileName || name,
        userID: req.user.id
      };

      if (req.body.pipeline) {
        // Not something we really need mongo's insight on
        jobToSubmit.inputQueryConfig.pipeline = JSON.stringify(
          req.body.pipeline
        );
      }

      // Carry over synonyms, if the user had dynamic synonyms saved
      if (job.search.synonyms) {
        jobToSubmit.search = {
          synonyms: job.search.synonyms
        };
      }

      if (req.body.options) {
        jobToSubmit.option = req.body.options;
      }

      Jobs.createUserJob(req, jobToSubmit, null, (err, newJob) => {
        newJob.submitAnnotation((err, submittedJob) => {
          if (err) {
            return res
              .status(400)
              .send(err.message)
              .end();
          }

          //log.debug("upload succeeded");
          // job saved, but not yet guaranteed to be submitted via redis
          return res.json(submittedJob).end();
        });
      });
    });
  }

  //TODO: maybe switch to job ID only
  // find.limit apparently faster than findOne https://codeandcodes.com/2014/07/31/mongodb-performance-enhancements-and-tweaks/
  //TODO: maybe switch to job ID only
  function getOne(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }

    Jobs.findById(req.params.id, projection)
      .lean()
      .populate("_creator", "_id name createdAt")
      .exec((err, job) => {
        if (err || !job) {
          return _tellUser(res, err, job);
        }

        if (!Jobs.isPermittedFor(req.user && req.user.id, job)) {
          return _notAuthorized(res);
        }

        //Save performance by not update unless we have to
        //If job is below safety threshold, give grace period
        if (job.expireDate < Date.now() + jobExpireSafety) {
          Jobs.collection.update(
            {
              _id: ObjectID(req.params.id)
            },
            {
              $set: {
                expireDate: new Date(Date.now() + jobExpireExtension)
              }
            },
            err => {
              if (err) {
                log.error(err);
              }
            }
          );
        }

        return _tellUser(res, err, job);
      });
  }

  function reIndex(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }

    Jobs.findById(req.params.id).exec((err, job) => {
      if (err || !job) {
        return _tellUser(res, err, job);
      }

      const owner = req.user && job.userID === req.user.id;

      if (!owner) {
        return _notAuthorized(res);
      }

      // remove the hidden fields
      Jobs.submitIndexJob(job, null, (err, uJob) => _tellUser(res, err, uJob));
    });
  }
  // function getOnePublic(req, res) {
  //   if (!req.params.id) { return _noJob(res); }

  //   Jobs.findOne({_id: req.params.id, visibility: 'public'}, {
  //     __v: 0, dirs: 0, 'inputQueryConfig.indexName': 0, 'inputQueryConfig.indexType': 0,
  //     'search.indexName': 0, 'search.indexType': 0,
  //   }, (err, job) => {
  //     _tellUser(res, err, job);
  //   });
  // }

  function deleteOne(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }
    if (!req.user) {
      return _notAuthorized(res);
    }

    // Need the indexName and indexType for deletion
    // TODO: fix that
    Jobs.findOne(
      {
        userID: req.user.id,
        _id: req.params.id
      },
      {
        __v: 0
      },
      (err, job) => {
        if (err || !job) {
          log.error("error deleting", err, job);
          return _tellUser(res, err, job);
        }

        job.deleteAnnotation((err, deletedJob) => {
          return _tellUser(res, err, deletedJob ? deletedJob.toJSON() : null);
        });
      }
    );
  }

  // TODO: better error handling

  function addSynonyms(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }
    if (!req.body.synonyms) {
      return _noJob(res);
    }
    if (!req.user) {
      return _notAuthorized(res);
    }

    // var synonyms = JSON.parse(req.query.synonyms);
    Jobs.findById(req.params.id, projection, (err, job) => {
      if (err || !job) {
        return _tellUser(res, err, job);
      }

      if (job.userID !== req.user.id) {
        return _notAuthorized(res);
      }
      // TODO: escape / don't trust input
      job.set("search.synonyms", req.body.synonyms);

      job.save((saveErr, savedJob) => {
        _tellUser(res, saveErr, savedJob);
      });
    });
  }

  function checkAnnotationStatus(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }

    // This is a bit of a security issue; any user could trigger refresh
    Jobs.findById(req.params.id, projection, (err, job) => {
      if (err || !job) {
        return _tellUser(res, err, job);
      }

      if (!Jobs.isPermittedFor(req.user && req.user.id, job)) {
        return _notAuthorized(res);
      }

      // Can't use lean; checkAnnotationStatus modifies the object potentially
      job.checkAnnotationStatus((err, job) => _tellUser(res, err, job));
    }).populate("_creator", "name _id createdAt");
  }

  function checkIndexStatus(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }

    // Can't use lean; checkAnnotationStatus modifies the object potentially
    // This is a bit of a security issue; any user could trigger refresh
    Jobs.checkIndexStatus(
      {
        _id: req.params.id
      },
      (err, job) => {
        if ((err && err !== "NOT_FOUND") || !job) {
          return _tellUser(res, err, job);
        }

        if (!Jobs.isPermittedFor(req.user && req.user.id, job)) {
          return _notAuthorized(res);
        }

        _tellUser(res, null, job);
      }
    );
  }

  // TODO: better error handling
  // find.limit apparently faster than findOne https://codeandcodes.com/2014/07/31/mongodb-performance-enhancements-and-tweaks/
  function searchJob(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }
    if (!req.body.searchBody) {
      return _noJob(res);
    }

    const query = req.body.searchBody;
    Jobs.collection.findOne(
      {
        _id: ObjectID(req.params.id)
      },
      {
        "search.indexName": 1,
        "search.indexType": 1,
        "search.queries": 1,
        visibility: 1,
        sharedWith: 1,
        userID: 1
      },
      (err, job) => {
        if (err || !job) {
          return _tellUser(res, err, job);
        }

        if (!Jobs.isPermittedFor(req.user && req.user.id, job)) {
          return _notAuthorized(res);
        }

        elasticClient
          .search({
            index: job.search.indexName,
            type: job.search.indexType,
            body: query
          })
          .then(
            response => {
              res.json(response);

              // if (owner) ...
              // const toSave = {
              //   date: Date.now(),
              //   action: Object.assign({}, req.query.searchBody),
              //   resultStatus: 'success',
              // };

              // if(job.search.queries.length <= Jobs.maxQueriesToSave) {
              //   job.search.queries.push(toSave);
              // } else {
              //   job.search.queries.shift();
              //   job.search.queries.push(toSave);
              // }

              // job.save( (err) => {
              //   if (err) {
              //     log.error(err);
              //   }

              // });
              // Save query
            },
            err => {
              if (err && err.body.error.type === "index_not_found_exception") {
                //TODO: Should we update the index status to with this error
                //or let user handlei t?
              }
              // TODO: improve error handling
              res.status(404).send(err.body.error.type);

              // const toSave = {
              //   date: Date.now(),
              //   action: req.query.searchBody,
              //   resultStatus: err.message,
              // };

              // if(job.search.queries.length <= Jobs.maxQueriesToSave) {
              //   job.search.queries.push(toSave);
              // } else {
              //   job.search.queries.shift();
              //   job.search.queries.push(toSave);
              // }

              // job.save( (err) => {
              //   if (err) {
              //     log.error(err);
              //   }
              // });
            }
          );
      }
    );
  }

  //TODO: finish
  function restart(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }
    if (!req.user) {
      return _notAuthorized(res);
    }

    Jobs.findOne(
      {
        userID: req.user.id,
        _id: req.params.id
      },
      (err, job) => {
        if (err || !job) {
          return _tellUser(res, err, job);
        }

        job.restart((err, updatedJob) => {
          _tellUser(res, err, updatedJob);
        });
      }
    );
  }

  // TODO: implement validation in Model to prevent funky overriding
  // Expects data to be sanitized by this point
  // TODO: only return requested fields to update; may need graphql
  function update(req, res) {
    if (!req.params.id) {
      return _noJob(res);
    }
    if (!req.user) {
      return _notAuthorized(res);
    }

    const fields = Object.keys(req.body);
    let err;
    Object.keys(projection).forEach(field => {
      if (field in req.body) {
        err = new Error("Protected field, won't modify");
        return;
      }
    });

    if (err) {
      return _tellUser(res, err);
    }

    Jobs.findOneAndUpdate(
      {
        userID: req.user.id,
        _id: req.params.id
      },
      req.body,
      {
        new: true,
        fields
      }
    )
      .lean()
      .exec((iErr, updatedJob) => {
        _tellUser(res, iErr, {
          job: updatedJob,
          paths: fields
        });
      });
  }
};

function _noJob(res, errMsg) {
  return _tellUser(res, errMsg, null);
}

function _tellUser(res, err, jobs) {
  if (err) {
    log.error(err);
    return res.status(errCode).send(err.message || err.msg || err);
  }

  if (!jobs) {
    return res.sendStatus(404);
  }

  return res.json(jobs); // let user handle truly empty sets
}

function _notAuthorized(res, job) {
  const err = new Error("You're not authorized to view this job");
  log.warn({
    err,
    job
  });
  return res.status(401).send(err.message);
}
