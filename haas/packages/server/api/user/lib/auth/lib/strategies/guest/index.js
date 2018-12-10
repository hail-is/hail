const express = require('express');
const passport = require('./passport');

module.exports = function initGuestRoutes(tokenManager) {
  const router = express.Router({mergeParams: true});

  router.post('/', function onPost(req, res, next) {
    passport.authenticate(function authCb(err, user) {
      if (err) { return res.json(401, err); }
      if (!user) {
        return res.json(404,
          {message: 'Something went wrong, please try again.'} );
      }
      tokenManager.sendTokenWithRefresh(user, null, res);
      next();
    });
  });

  return router;
};
