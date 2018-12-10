const express = require("express");
const Downloader = require("./download");
const Uploader = require("./upload");
const controller = require("./controller");

class JobRouter {
  /* @param {Obj} config Job config
   * @param {Obj} comm Job Comm instance
   * @param {Obj} Model Mongoose model
   */
  constructor(jModel, userInstance, jobConfig, jobComm) {
    // expect that req[userProp] is available to all routers
    const uM = userInstance.middleware;
    const userModel = userInstance.Model;

    const downloader = new Downloader(jModel, uM, jobConfig);
    const uploader = new Uploader(jModel, jobComm, userModel, uM);

    const router = express.Router({ mergeParams: true });

    router.use("/download", downloader.router);
    router.use("/upload", uploader.router);

    const routeCtrl = controller(jModel, jobComm);

    // WARNING: Requires check for if(!req.user) for any truly protected routs
    // Requires careful granular control
    router.use("/", uM.verifyToken, this.protectedRoutes(routeCtrl));

    this.router = router;
  }

  publicRoutes(routeCtrl) {
    const router = express.Router({ mergeParams: true });
    const rC = routeCtrl;
    // get all unfinished jobs
    // router.get('/public', rC.getPublic);
    // router.get('/public/:id', rC.getOnePublic);
    // router.get('/public/search/:id', rC.getOnePublic);

    return router;
  }

  protectedRoutes(routeCtrl) {
    const router = express.Router({ mergeParams: true });
    const rC = routeCtrl;
    // get all  completed jobs
    // router.get('/', rC.getComplete);

    router.get("/list/:type/:visibility?", rC.getJobs);
    // get all unfinished jobs
    // router.get('/incomplete', rC.getIncomplete);
    // get all config
    router.get("/config/:assembly", rC.getConfig);

    // router.get('/failed', rC.getFailed);
    // get a finished job
    router.param("/:id", rC.getOne);

    router.get("/:id/annotationStatus", rC.checkAnnotationStatus);

    router.get("/:id/indexStatus", rC.checkIndexStatus);
    // update a job
    // strangely when this is 'put', I get unathorized error
    // (when mistakenly sending a upload request here)
    router.delete("/:id", rC.deleteOne);

    //todo: change to :id/search
    // router.get('/searchJob/:id', rC.searchJob);
    router.post("/:id/search", rC.searchJob);

    // General update
    router.post("/:id", rC.update);

    router.post("/:id/saveFromQuery/", rC.saveFromQuery);

    router.post("/:id/addSynonyms/", rC.addSynonyms);

    router.post("/:id/reIndex", rC.reIndex);
    // may seem like this should start the job; however that's coupled to uploading
    // and so is handled in the upload.js controller
    // router.post('/:id', rC.restart);
    // will go to uploader router.post('/new/', rC.create);

    return router;
  }
}

module.exports = JobRouter;
