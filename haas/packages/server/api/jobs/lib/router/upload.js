/*eslint-disable class-methods-use-this*/

const express = require('express');

const log = require.main.require('./common/logger');

const PromiseBird = require('bluebird');

const Busboy = require('busboy');

const path = require('path');

const fs = PromiseBird.promisifyAll(require('fs-extra'));

const AWS = require('aws-sdk');

const tempDir = process.env.TEMP_DIR;

if (!tempDir) {
  throw new Error('TEMP_DIR env constiable required');
}

const s3Bucket = process.env.S3_BUCKET;

// TODO: harden against errors
class Uploader {
  constructor(JobsModel, jobComm, UserModel, userAuth) {
    this.Jobs = JobsModel;
    this.jobComm = jobComm;

    this.User = UserModel;
    this.streamUpload = this.streamUpload.bind(this);
    this.uploadToS3 = this.uploadToS3.bind(this);
    this.router = this.initRouter(
      userAuth.verifyTokenPermissive({ credentialsRequired: true })
    );
  }

  initRouter(tokenAuth) {
    const router = express.Router({ mergeParams: true });
    // auth.verifyToken verifies the user's authenticity
    // TODO: support non-HTML5 browsers
    router.post('/', tokenAuth, this.streamUpload);

    // Upload to some service
    router.post(
      '/:id',
      (req, res, next) => {
        res.connection.setTimeout(0);
        next();
      },
      tokenAuth,
      this.uploadToS3
    );

    return router;
  }

  // expects a token instance method on Job models
  streamUpload(req, res) {
    if (req.body.file && req.body.job) {
      if (req.body.file.type === 's3') {
        this.createFromS3(req, res);
      }
    } else {
      this.createFromFileToS3(req, res);
    }
  }

  createFromFileToS3(req, res) {
    const { user } = req;

    if (!user && user.id) {
      res.status(401).end();
      return;
    }

    const busboy = new Busboy({ headers: req.headers });

    let jobObj;

    busboy.on('field', (fieldName, val) => {
      if (fieldName === 'job') {
        jobObj = JSON.parse(val);
      }

      console.info('GOT JOB OBJ', jobObj);
    });

    let inputFileName;

    busboy.on('file', (fieldname, file, filename, encoding, mimetype) => {
      console.log(
        `File [${fieldname}]: filename: ${filename}, encoding: 
          ${encoding}, mimetype: ${mimetype}
        `
      );

      const s3 = new AWS.S3({
        params: { Bucket: s3Bucket, Key: filename, Body: file },
        options: { partSize: 5 * 1024 * 1024, queueSize: 10 } // 5 MB
      });

      inputFileName = filename;

      // TODO: Check that it doesn't exist
      // TODO: handle upload failure
      s3.upload()
        .on(
          'httpUploadProgress',
          evt => console.info('progress', evt, jobObj)
          // this.jobComm.notifyUser("httpUploadProgress", job, evt)
        )
        // It is finished here
        .send((err, data) => {
          if (err) {
            log.error(err);
          }
          console.log('done', err, data);
        });
    });

    busboy.on('finish', () => res.status(200).end());
    req.pipe(busboy);
  }

  createFromS3(req, res) {
    if (!req.user) {
      res.status(401).end();
      return;
    }

    const filename = req.body.file.name;
    const jobObj = req.body.job;

    console.info('CREATING FROM S3');
    this.submitJobIfComplete(jobObj, filename, req.user, res);
  }

  submitJobIfComplete(jobObj, inputFileName, user, res) {
    const parts = inputFileName.split('.');
    const extension = parts.slice(1, parts.length).join('.');
    const isFam = !!extension.match('.fam');

    let toSet = {};
    if (isFam) {
      toSet = { 'inputFiles.fam': inputFileName };
    } else {
      toSet = { 'inputFiles.data': inputFileName };
    }

    console.info(
      'Starting to create or update',
      inputFileName,
      new Date().getTime()
    );

    this.Jobs.findOneAndUpdate(
      { name: jobObj.name },
      { $set: toSet },
      { new: true }
    )
      .then(job => {
        if (!job) {
          console.info('No job existed', inputFileName, new Date().getTime());

          jobObj.outputBaseFileName = `${jobObj.name}.results.tsv.gz`;

          if (isFam) {
            jobObj.inputFiles = {
              fam: inputFileName
            };
          } else {
            jobObj.inputFiles = {
              data: inputFileName
            };
          }

          const tJob = this.Jobs.createUserJob(jobObj, user);

          return tJob.save(); //returns a promise;
        }

        console.info('A job existed', inputFileName, new Date().getTime());
        return job;
      })
      .then(job => {
        console.info('Saved or updated', inputFileName, new Date().getTime());

        if (job.inputFiles.fam && job.inputFiles.data) {
          console.info('Submitting', inputFileName, new Date().getTime());

          // Promisifying doesn't work, "this" becomes undefined
          job.submitAnnotation((err, savedJob) => {
            if (err) {
              res.status(500).send("Couldn't submit job");
            }

            res.status(200).json(savedJob);
          });

          return;
        }

        res.status(200).end();
      })
      .catch(err => {
        // throws for some reason with no error
        // mongoose has not good support for promises
        if (err) {
          log.eror(err);

          res.status(500).send("Uhoh! We have a database issue, we're on it!");
        }
      });
  }

  // uploadFromFile(req, res) {
  //   if (!req.user) {
  //     return res.sendStatus(401);
  //   }

  //   let jobObj;

  //   const busboy = new Busboy({ headers: req.headers });
  //   busboy.on("field", (fieldName, val) => {
  //     if (fieldName === "job") {
  //       jobObj = JSON.parse(val);
  //     }
  //   });

  //   // constiables that will be used in finishing the upload.
  //   let tempUploadPath;
  //   let fileName;

  //   busboy.on("file", (fieldName, file, tFilename) => {
  //     fileName = tFilename;
  //     // TODO: Can we just save the file directly to the right place
  //     tempUploadPath = path.join(tempDir, path.basename(fileName));

  //     const fstream = fs.createWriteStream(tempUploadPath);
  //     file.pipe(fstream);
  //     fstream.on("close", () => {});
  //   });

  //   busboy.on("error", err => {
  //     log.error(err);

  //     return res.status(500).send("Failed to save input file");
  //   });

  //   busboy.on("finish", () => {
  //     // write test to expect createdDate to equal Mongoose createdDate
  //     const unsavedJob = this.Jobs.createUserJob(jobObj, user);

  //     // Guaranteed to have job here

  //     const inFilePath = path.join(
  //       unsavedJob.dirs.in,
  //       unsavedJob.inputFileName
  //     );

  //     return fs
  //       .moveAsync(tempUploadPath, inFilePath, { clobber: 1 })
  //       .catch(errMove => {
  //         log.error(errMove);
  //         return res
  //           .status(500)
  //           .send("Failed to move input file to final destination");
  //       })
  //       .then(() => {
  //         unsavedJob.submitAnnotation((submitErr, submittedJob) => {
  //           if (submitErr) {
  //             log.error(submitErr);
  //             return res.status(500).send("Failed to save job");
  //           }

  //           // log.debug("upload succeeded");
  //           // job saved, but not yet guaranteed to be submitted via redis
  //           return res.json(submittedJob).end();
  //         });
  //       });
  //   });
  //   req.pipe(busboy);
  // }

  // uploadFromS3(req, res) {
  //   if (!req.user) {
  //     return res.sendStatus(401);
  //   }

  //   const fileData = req.body.file;
  //   const jobData = req.body.job;

  //   let cancelled;

  //   if (!fileData) {
  //     return res.status(404).send("File missing");
  //   }

  //   if (!jobData) {
  //     return res.status(404).send("Job data missing");
  //   }

  //   return this.User.findById(req.user.id, { cloud: 1 }, (err, user) => {
  //     if (err) {
  //       log.error(err);
  //       return res.sendStatus(500);
  //     }

  //     if (!user) {
  //       log.error(new Error(`no user found for id ${req.user.id}`));
  //       return res.status(404).send("No user found");
  //     }

  //     const { accessID, secret } = user.cloud.s3.credentials;

  //     if (!(accessID && secret)) {
  //       return res.status(404).send("No S3 credentials found");
  //     }

  //     const s3obj = new AWS.S3({
  //       accessKeyId: accessID,
  //       secretAccessKey: secret
  //     });

  //     const tempUploadPath = path.join(tempDir, path.basename(fileData.name));
  //     const file = fs.createWriteStream(tempUploadPath);

  //     let accumulatedSize = 0;
  //     const totalSize = fileData.Size;

  //     // If using pipe and a readable stream, put the submitAnnotation code here
  //     // file.on('finish', () => {
  //     // move to final destination and submitAnnotation
  //     // })

  //     let last;

  //     // Thanks http://stackoverflow.com/questions/33883324/node-aws-s3-getobject-firing-httpdone-before-file-stream-completes
  //     // https://github.com/aws/aws-sdk-js/issues/744
  //     const s3GetObjectRequest = s3obj.getObject({
  //       Bucket: fileData.Bucket,
  //       Key: fileData.Key
  //     });

  //     req.connection.on("close", () => {
  //       cancelled = true;
  //       s3GetObjectRequest.abort();
  //     });

  //     // TODO: Better error handling
  //     s3GetObjectRequest
  //       .createReadStream()
  //       .on("error", streamErr => {
  //         if (streamErr.code !== "RequestAbortedError") {
  //           log.error(err);
  //           return res.status(500).send(err.code);
  //         }

  //         return res.status(200);
  //       })
  //       .on("data", chunk => {
  //         const now = Date.now();

  //         accumulatedSize += chunk.length;

  //         if (!last || now >= last + 250) {
  //           this.jobComm.notifyUser(
  //             "httpDownloadProgress",
  //             // identifier
  //             {
  //               _id: -9,
  //               userID: req.user.id
  //             },
  //             // data
  //             {
  //               name: fileData.Key,
  //               loaded: accumulatedSize,
  //               total: totalSize
  //             }
  //           );

  //           last = now;
  //         }
  //       })
  //       .on("finish", () => {
  //         // Do we need this?
  //       })
  //       .pipe(file);

  //     return file.on("finish", () => {
  //       const fileName = fileData.name;

  //       if (cancelled) {
  //         return res.sendStatus(200);
  //       }

  //       return this.Jobs.createUserJob(
  //         req,
  //         jobData,
  //         fileName,
  //         (createErr, unsavedJob) => {
  //           if (createErr) {
  //             return res.status(500).send("Failed to create job");
  //           }

  //           const inFilePath = path.join(
  //             unsavedJob.dirs.in,
  //             unsavedJob.inputFileName
  //           );

  //           return fs
  //             .moveAsync(tempUploadPath, inFilePath, { clobber: 1 })
  //             .catch(moveErr => {
  //               log.error(moveErr);
  //               return res
  //                 .status(500)
  //                 .send("Failed to move file to final destination");
  //             })
  //             .then(() =>
  //               unsavedJob.submitAnnotation((submitErr, submittedJob) => {
  //                 if (submitErr) {
  //                   log.error(submitErr);
  //                   return res.status(500).send(submitErr.message);
  //                 }

  //                 // job saved, but not yet guaranteed to be submitted via redis
  //                 return res.json(submittedJob).end();
  //               })
  //             );
  //         }
  //       );
  //     });

  //     //Can also use httpDownloadProgress event
  //   });
  // }

  uploadToS3(req, res) {
    if (!req.user) {
      return res.sendStatus(401);
    }

    if (!req.params.id) {
      return res.status(400).send('Unknown job id');
    }

    const bucket = req.body.Bucket;

    // optional: will take the file name by default
    let key = req.body.Key;

    if (!bucket) {
      return res.status(400).send('Missing Bucket');
    }

    return this.User.findById(req.user.id, { cloud: 1 }, (err, user) => {
      if (err) {
        log.error(err);
        return res.sendStatus(500);
      }

      if (!user) {
        log.error(new Error(`no user found for id ${req.user.id}`));
        return res.status(404).send('No user found');
      }

      const { accessID, secret } = user.cloud.s3.credentials;

      if (!(accessID && secret)) {
        return res.status(404).send('No S3 credentials found');
      }

      // Can't use lean; checkAnnotationStatus modifies the object potentially
      return this.Jobs.findById(req.params.id, (findErr, job) => {
        if (findErr) {
          log.error(findErr);
          return res.status(500).send("Couldn't retrieve this annotation");
        }

        if (!job) {
          return res.sendStatus(404).send('Job not found');
        }

        if (!this.Jobs.isPermittedFor(user._id, job)) {
          return res.sendStatus(401);
        }

        const file =
          job.outputFileNames.archived || job.outputFileNames.compressed;

        if (!file) {
          return res.status(404).send('Job output file not found');
        }

        key = key || file;

        const filePath = path.join(job.dirs.out, file);

        const readStream = fs.createReadStream(filePath);

        const s3obj = new AWS.S3({
          accessKeyId: accessID,
          secretAccessKey: secret
        });

        const s3request = s3obj.upload({
          Bucket: bucket,
          Key: key,
          Body: readStream
        });

        req.connection.on('close', () => {
          s3request.abort();
        });

        job.userID = user._id;

        return s3request
          .on('httpUploadProgress', evt => {
            this.jobComm.notifyUser('httpUploadProgress', job, evt);
          })
          .send((sendErr, data) => {
            if (sendErr && sendErr.code !== 'RequestAbortedError') {
              log.error(sendErr);
              return res.status(500).send(err.code);
            }

            // TODO: figure out if there's any way to avoid using this
            // Using because it seems on occassion s3 will send us here twice
            if (!res.headersSent) {
              return res.sendStatus(200);
            }
          });
      });
    });
  }
}

module.exports = Uploader;
