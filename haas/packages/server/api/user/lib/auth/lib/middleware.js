const expressJwt = require('express-jwt');

const InvalidTokenError = require.main.require(
  './common/auth/errors/InvalidTokenError'
);

// const log = require.main.require('./common/logger');

const _ = require('lodash');

// const jwksRsa = require('jwks-rsa');

// Create middleware for checking the JWT
// const checkJwt = jwt({
//   // Dynamically provide a signing key based on the kid in the header and the singing keys provided by the JWKS endpoint.
//   secret: jwksRsa.expressJwtSecret({
//     cache: true,
//     rateLimit: true,
//     jwksRequestsPerMinute: 5,
//     jwksUri: process.env.AUTH0_WEB_KEY_SET_URL
//   }),

//   // Validate the audience and the issuer.
//   audience: process.env.AUTH0_AUDIENCE,
//   issuer: process.env.AUTH0_DOMAIN,
//   algorithms: ['RS256']
// });

class AuthMiddleware {
  constructor(User, tokenManager) {
    this.User = User;

    this.tokenMan = tokenManager;

    this.verifyToken = expressJwt({
      userProperty: tokenManager.attachProperty,
      secret: tokenManager.secret,
      getToken: tokenManager.getAccessTokenFromRequest,
      // This isn't exactly right, the access token will contain very little
      // user-specific code, just the sub and thes scope array
      requestProperty: 'user',
      audience: process.env.AUTH0_AUDIENCE,
      issuer: process.env.AUTH0_DOMAIN,
      algorithms: ['RS256'],
      credentialsRequired: false
    });

    // elase not guaranteed to respect this
    this.hasRole = this.hasRole.bind(this);
    this.verifyTokenPermissive = this.verifyTokenPermissive.bind(this);
  }

  hasRole(requiredRole) {
    // TODO: do we need to return closure. write test
    return function hasRoleClosure(req, res, next) {
      if (!this.User.hasRole(req.user, requiredRole)) {
        return res.send(403);
      }
      next();
    };
  }

  // like verify token, but will check if the user submitted a valid refresh token
  // this is useful in the case the user hasn't had a chance to refresh their
  // id_token, but still has a valid refresh token
  verifyTokenPermissive(opts = { credentialsRequired: true }) {
    return (req, res, next) => {
      // const accessToken = this.tokenMan.getAccessTokenFromRequest(req);

      // if (!accessToken) {
      //   if (!opts.credentialsRequired) {
      //     next();
      //     return;
      //   }

      //   next(
      //     new InvalidTokenError('credentials_required', {
      //       message: 'No access token was found'
      //     })
      //   );

      //   return;
      // }

      return this.verifyToken(req, res, next);
      // this.verifyToken(req, res, (err, user) => {
      //   if (err) {
      //     next(err, null);
      //     return;
      //   }

      //   _.set(req, this.tokenMan.attachProperty, user);

      //   next();
      // });
    };
  }
  // At the moment we do not have a refresh token
  // const refreshToken = this.tokenMan.getRefreshTokenFromRequest(req);

  // this.tokenMan.refreshTokens(
  //   idToken,
  //   refreshToken,
  //   (err, newIdToken, newRefreshToken, userData) => {
  //     if (err) {
  //       if (!newRefreshToken) {
  //         // We have an invalid refresh token, so clear it
  //         this.tokenMan.clearRefreshTokenFromHeader(res);
  //       }

  //       next(err);
  //       return;
  //     }

  //     if (refreshToken != newRefreshToken) {
  //       this.tokenMan.setRefreshTokenCookie(newRefreshToken, res);
  //     }

  //     _.set(req, this.tokenMan.attachProperty, userData);
  //     next();
  //   }
  // );
  // };
  // }
}

module.exports = AuthMiddleware;
