const express = require('express');
const passport = require('passport');
const log = require.main.require('./common/logger');

/* expects 'token' instance method on User models*/
module.exports = function initLocalRoutes(tokenManager) {
  const router = express.Router();

  router.post('/', function postCb(req, res) {
    passport.authenticate('local', function authCb(err, user) {
      if (err) {
        log.warn(err);
        return res.sendStatus(500);
      }

      if (!user) {
        return res.sendStatus(422);
      }

      const accessToken = tokenManager.signAccessToken(user.token);

      if(!accessToken) {
        throw new Error("No access token created");
      }

      tokenManager.sendTokenWithRefresh(accessToken, user.refreshToken, res);
      return;
    })(req, res);
  });

  return router;
};
