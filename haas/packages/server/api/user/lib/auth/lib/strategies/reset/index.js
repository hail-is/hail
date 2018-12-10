const express = require('express');
const passport = require('./passport');
const log = require.main.require('./common/logger');

module.exports = function initGuestRoutes(tokenManager, auth) {
  const router = express.Router({mergeParams: true});

  router.get('/', function onGet(req, res) {
    passport.sendResetLink(req, function(err, result) {
      if (err) {
        log.error(err);
        return res.sendStatus(401);
      }
      if (!result) {
        return res.sendStatus(422);
      }
      // don't tell the user anything was wrong if no email found
      return res.sendStatus(200);
    });
  });

  router.post('/', function onPost(req, res) {
    passport.resetPassword(req, function(err, result) {
      if (err) {
        log.error(err);
        return res.sendStatus(401);
      }
      if (!result) {
        return res.sendStatus(422);
      }
      return res.sendStatus(200);
    });
  });

  return router;
};
