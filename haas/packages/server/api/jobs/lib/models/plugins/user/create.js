const PromiseBird = require('bluebird');
const Busboy = require('busboy');
const path = require('path');
const fs = PromiseBird.promisifyAll(require('fs-extra'));
const os = require('os');
const _ = require('lodash');
const log = require.main.require('./common/logger');

const tempDir = process.env.TEMP_DIR;

if (!tempDir) {
  throw new Error('TEMP_DIR env variable required');
}

exports = module.exports = function userJobCreationPlugin(schema, user) {
  // user methods
  const _getUser = user.getUser;
  const _isGuest = user.isGuest;
  const _getUserId = user.getUserId;

  const _guestExpiration = 4838400e3; // 2 months;
  const _inFolder = 'input';
  const _outFolder = 'output';
  const _jobProp = 'job';

  schema.statics.createUserJob = function createUserJob(
    request,
    jobObj,
    fileName,
    cb
  ) {
    //This is 32
    const Model = this;

    const jobToSubmit = Object.assign({}, jobObj);

    _getUser(request, (err, user) => {
      const userID = _getUserId(user);

      if (!userID) {
        throw new Error("no userID found, job can't be saved");
      }

      // We want to let Mongo set the _id
      delete jobToSubmit._id;
      delete jobToSubmit.id;

      const jobDoc = new Model(jobToSubmit);

      jobDoc.userID = userID;

      if (_isGuest(user)) {
        jobDoc.expireDate = Date.now() + _guestExpiration;
      }

      let subDirName = '';
      let inputBaseName = '';

      // If no input file is provided, we have no need to make an input directory
      if (fileName) {
        if (fileName.indexOf(' ')) {
          fileName = fileName.split(' ').join('_');
        }

        subDirName = path.dirname(fileName);
        inputBaseName = path.basename(fileName);

        jobDoc.inputFileName = inputBaseName;
        jobDoc.dirs.in = path.join(
          jobDoc.baseDir,
          userID,
          jobDoc._id.toString(),
          _inFolder,
          subDirName
        );

        try {
          fs.ensureDirSync(jobDoc.dirs.in);
        } catch (ensureErr) {
          log.error(ensureErr);
          return cb(ensureErr, null);
        }
      } else {
        delete jobDoc.inputFileName;
        delete jobDoc.dirs.in;
      }

      jobDoc.dirs.out = path.join(
        jobDoc.baseDir,
        userID,
        jobDoc._id.toString(),
        _outFolder,
        subDirName
      );

      try {
        process.umask(0);
        fs.ensureDirSync(jobDoc.dirs.out, { mode: parseInt('0777', 8) });
      } catch (ensureErr) {
        log.error(ensureErr);
        return cb(ensureErr, null);
      }

      // use our own extension in the output path
      // this avoids ambiguous paths like temp.zip.annotated
      // however, I don't want to change the file too much; so I expect
      // that the last string after a period is an extension, which we will modify
      if (!jobDoc.name) {
        if (inputBaseName) {
          jobDoc.name = inputBaseName.substr(0, inputBaseName.lastIndexOf('.'));
        } else {
          const err = new Error('Job name required');
          log.error(err);
          return cb(err, null);
        }
      }

      // The annotator will append various extensions to this path
      // So this just specifies the output directory and the files' prefixes
      jobDoc.outputBaseFileName = jobDoc.outputBaseFileName || jobDoc.name;

      return cb(null, jobDoc);
    });
  };

  schema.statics.upload = function upload(req, save, tCb) {
    const cb = (_.isFunction(tCb) && tCb) || function noop() {};

    const self = this;

    let jobObj;

    const busboy = new Busboy({ headers: req.headers });
    busboy.on('field', (fieldName, val) => {
      if (fieldName === _jobProp) {
        jobObj = JSON.parse(val);
      }
    });

    // Variables that will be used in finishing the upload.
    let tempUploadPath;
    let filename;

    busboy.on('file', (fieldName, file, tFilename) => {
      filename = tFilename;
      // TODO: Can we just save the file directly to the right place

      tempUploadPath = path.join(tempDir, path.basename(filename));

      const fstream = fs.createWriteStream(tempUploadPath);
      file.pipe(fstream);
      fstream.on('close', () => log.debug('busboy closed the stream'));
    });

    busboy.on('error', err => {
      log.error(err);
      return cb(err);
    });

    busboy.on('finish', () => {
      // write test to expect createdDate to equal Mongoose createdDate
      self.createUserJob(req, jobObj, (err, job) => {
        if (err) {
          cb(err);
          return;
        }

        const inFilePath = path.join(job.dirs.in, filename);

        fs.moveAsync(tempUploadPath, inFilePath, { clobber: 1 })
          .catch(err => {
            log.error(err);
            cb(err);
          })
          .then(() => {
            // use our own extension in the output path
            // this avoids ambiguous paths like temp.zip.annotated
            // however, I don't want to change the file too much; so I expect
            // that the last string after a period is an extension, which we will modify
            if (!job.name) {
              job.name = filename.substr(0, filename.lastIndexOf('.'));
            }

            job.inputFileName = filename;

            // The annotator will append various extensions to this path
            // So this just specifies the output directory and the files' prefixes
            job.outputBaseFileName = job.outputBaseFileName || job.name;

            return cb(null, job);
          });
      });
    });
    req.pipe(busboy);
  };
};

/*After upload to temp folder is finished, generates destinationFilePath, and moves file to it, record to session
*@params: req.body.inputType
*req.body.jobDate should be null for uploads in order to allow server to handle those; in case where multiple subsequent uploads leading to mismatched job numbers
*@returns: void
*side-effects: (str) destinationFilePath, move file from tmp to destinationFilePath, create unfinishedJobs key in session (if not exit),
*        create Date() jobDate, store jobDate to session.unfinishedJobs[req.body.inputType] (this is set to null at the end of the job lifecycle,
  store session[req.body.inputType][jobDate] = (obj) { 'inputType' : req.body.inputType } as a longer-standing record of job
  emit to user room (based on session.id) uploadComplete event w/ jobDate, inputType, and fileType
*/
