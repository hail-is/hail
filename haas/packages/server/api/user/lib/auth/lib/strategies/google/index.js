const express = require('express');
const passport = require('passport');

module.exports = function initGoogleRoutes(authUtil) {
  const router = express.Router({mergeParams: true});

  router
  .get('/', passport.authenticate('google', {
    failureRedirect: '/signup',
    scope: [
      'https://www.googleapis.com/auth/userinfo.profile',
      'https://www.googleapis.com/auth/userinfo.email',
    ],
    session: false,
  }))

  .get('/callback', passport.authenticate('google', {
    failureRedirect: '/signup',
    session: false,
  }), authUtil.setTokenCookie);

  return router;
};
