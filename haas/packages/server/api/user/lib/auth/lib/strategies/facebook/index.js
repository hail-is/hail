const express = require('express');
const passport = require('passport');

// TODO: need to implement token Cookie setter for facebook, google, twitter
module.exports = function initFacebookRoutes(authUtil) {
  const router = express.Router({mergeParams: true});

  router
  .get('/', passport.authenticate('facebook', {
    scope: ['email', 'user_about_me'],
    failureRedirect: '/signup',
    session: false,
  }))

  .get('/callback', passport.authenticate('facebook', {
    failureRedirect: '/signup',
    session: false,
  }), authUtil.setTokenCookie);

  return router;
};
