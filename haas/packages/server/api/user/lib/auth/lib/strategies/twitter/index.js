const express = require('express');
const passport = require('passport');

module.exports = function initTwitterRoutes(authUtil) {
  const router = express.Router({mergeParams: true});
  
  router
  .get('/', passport.authenticate('twitter', {
    failureRedirect: '/signup',
    session: false,
  }))

  .get('/callback', passport.authenticate('twitter',
  {
    failureRedirect: '/signup',
    session: false,
  }), authUtil.setTokenCookie);

  return router;
};
