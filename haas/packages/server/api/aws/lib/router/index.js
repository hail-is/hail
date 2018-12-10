const express = require('express');
const controller = require('./controller');

class AWSrouter {
  /* @param {Obj} config aws config
   * @param {Obj} comm Job Comm instance
   * @param {Obj} Model Mongoose model
  */
  constructor(userInstance, config) {
    // expect that req[userProp] is available to all routers
    const uM = userInstance.middleware;
    const User = userInstance.Model;

    const router = express.Router({mergeParams: true});

    const routeCtrl = controller(User);

    const middleware = uM.verifyTokenPermissive({credentialsRequired: true});

    router.get('/bucket', middleware, routeCtrl.listS3Buckets);
    router.get('/bucket/:bucketName', middleware, routeCtrl.listS3bucketContents);
    router.post('/s3signature', middleware, routeCtrl.createS3signature);

    this.router = router;
  }
}

module.exports = AWSrouter;
