const express = require('express');

class CIrouter {
  /* @param {Obj} config aws config
   * @param {Obj} comm Job Comm instance
   * @param {Obj} Model Mongoose model
   */
  constructor(userInstance, config) {
    // expect that req[userProp] is available to all routers
    const uM = userInstance.middleware;

    const router = express.Router({ mergeParams: true });

    // TODO: This is a generic "get access token function"
    // Move it to middleware
    router.get(
      '/',
      // adds user to request object
      uM.verifyToken,
      //adds accessToken poperty to request object
      uM.getAuth0ProviderAccessToken,

      async (req, res) => {
        console.info(req.accessToken);
        res.sendStatus(200);
      }
    );

    this.router = router;
  }
}

module.exports = CIrouter;
