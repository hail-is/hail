const express = require("express");
const PromiseBird = require("bluebird"); // Promise native to es
const jwt = require("jsonwebtoken");
const path = require("path");
const mime = require("mime");
const fs = PromiseBird.promisifyAll(require("fs-extra"));
const glob = PromiseBird.promisifyAll(require("glob"));
const log = require.main.require("./common/logger");

class Downloader {
  constructor(JobsModel, userAuth, config, comm) {
    this.Jobs = JobsModel;
    this.comm = comm;
    this.secret = config.download.secret;
    this.streamDownload = this.streamDownload.bind(this);
    this.sendToken = this.sendToken.bind(this);

    // Careful security vetting needed, since credentials not required
    const router = express.Router({ mergeParams: true });

    router.get("/:id", userAuth.verifyToken, this.sendToken);

    this.router = router;
  }

  initRouter(tokenAuth) {
    const router = express.Router({ mergeParams: true });

    // router.post("/token/:id", tokenAuth, this.sendToken);

    return router;
  }

  /* expects a token instance method on Job models*/
  sendToken(req, res) {
    if (!req.params.id) {
      const err = new Error("id required in request");
      log.error(err);
      return res
        .status(404)
        .send(err.message)
        .end();
    }

    this.Jobs.findById(req.params.id, (err, job) => {
      if (err) {
        log.error(err);
        return res
          .status(404)
          .send(err.message)
          .end();
      }

      if (!job) {
        return res.sendStatus(404);
      }

      if (!job.isPermittedFor(req.user && req.user.id)) {
        return res.sendStatus(401);
      }

      const token = jwt.sign({ id: job._id }, this.secret, { expiresIn: 60 });

      res
        .status(200)
        .send(token)
        .end();
    });
  }

  streamDownload(req, res) {
    const id = req.params.id;
    log.info("id", id);

    // jwt.verify(token, this.secret, (vErr, decoded) => {
    //   if (vErr) {
    //     return res.status(401).send(vErr);
    //   }

    //const fileNameKey = req.params.fileNameKey;
    // let fileName = decoded.fileName;

    this.Jobs.findOne({ _id: id }, (err, job) => {
      if (err) {
        log.error(err);
        return res.status(404).send(err);
      }

      const fileName =
        job.outputFileNames.archived || job.outputFileNames.compressed;

      console.info("fileName is", fileName);

      const filePath = path.join(job.dirs.out, fileName);

      try {
        const stats = fs.lstatSync(filePath);

        console.info("stats", stats, stats.isFile());
        if (!stats.isFile()) {
          return res.status(404).send("File not found");
        }
      } catch (e) {
        log.error(e);
        console.info("error", e);
        return res.status(404).send(e.message);
      }

      console.info("no error");
      _createDownloadStream(filePath, res);
    });
    // });
  }
}

exports = module.exports = Downloader;
/* private*/
function _createDownloadStream(filePath, res) {
  try {
    if (!filePath || !res) {
      log.error("no filePath or res passed to createDownloadStream");
    }

    const fileName = path.basename(filePath);

    const mimetype = mime.getType(filePath);

    res.setHeader("Content-disposition", "attachment; filename=" + fileName);

    /* needed for jquery.fileDownload*/
    res.cookie("fileDownload", "true");

    /* needed for jquery.fileDownload*/
    res.setHeader("Cache-Control", "max-age=60, must-revalidate");

    res.setHeader("Content-type", mimetype);

    const stream = fs.createReadStream(filePath);

    stream.on("error", function errCb(err) {
      // TODO: handle error
      log.error(err);
      res.clearCookie("fileDownload");

      return res.status(404).send(err);
      // handleError(err, res, true);
    });

    stream.pipe(res);
  } catch (err) {
    // TODO: handle error
    log.error(err);
    res.clearCookie("fileDownload");

    return res.status(404).send(err);
    // handleError(err,res,true);
  }
}

//TODO: use more robust zip file finding function
// function _findOrUpdatePath(filePath, cb) {
//   glob(`${path.dirname(filePath)}/*.gz`, null, function(er, files) {
//     if (er) { return cb(er); }
//     if (files.length) {
//       if (files.length > 1) {
//         log.warn('More than one gz file found in outputFilePath, using first', files);
//       }
//       cb(null, files[0]);
//     } else {
//       // default to the original file path, we expect it to exist
//       // but if it doesn't an error will be generated on stream, which is ok
//       // saves time over globbing yet again, same result
//       cb(null, filePath);
//     }
//   });
// }
