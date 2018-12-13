const express = require('express');

const clearCacheMessage = "Did you clear your cache? Missing user session.\nPlease log in again";
module.exports = function initRefreshRoute(tokenMan) {
  const router = express.Router({mergeParams: true});

  router.get('/', function onGet(req, res) {
    const maybeExpiredToken = tokenMan.getAccessTokenFromRequest(req);

    if (!maybeExpiredToken) {
      return res.status(401).send(clearCacheMessage);
    }

    const maybeRefreshToken = tokenMan.getRefreshTokenFromRequest(req);

    if (!maybeRefreshToken) {
      return res.status(401).send(clearCacheMessage);
    }

    tokenMan.refreshTokens(maybeExpiredToken, maybeRefreshToken, true, (err, idToken, refreshToken) => {
      if(err) {
        // refreshToken could have been invalid, and cleared from db
        // Note that we will also clear the refresh token if the user supplies
        // us a bogus one
        let message;
        if(err.code === 'invalid_refresh_token') {
          message = 
          tokenMan.clearRefreshTokenFromHeader(res);
        }

        
        if(err.code === 'mismatched_refresh_token') {
          message = "It looks like you\'re logged in on another device.\nPlease log in again.";
        } else if (err.code === 'missing_refresh_token' || err.code === 'missing_token') {
          message = clearCacheMessage;
        } else if (err.code === 'invalid_token' || err.code === 'invalid_refresh_token') {
          message = "You session looks odd.\nPlease log in again, and email us with any concerns";

          // TODO: Email admin here
        }
        // We may have failed to update the token in the db
        // TODO: clear the user? Shouldn't matter, we can't trust the client
        res.status(err.status).send(message);
  
        return;
      }

      // Future refreshTokens api may rehydrate the refresh token
      // should we choose to expire those
      // Can this ever happen?
      if(maybeRefreshToken && refreshToken && maybeRefreshToken != refreshToken) {
        tokenMan.sendTokenWithRefresh(idToken, refreshToken, res);
      } else {
        tokenMan.sendToken(idToken, res);
      }
      
    });
  });

  return router;
};
