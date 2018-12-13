const TokenManager = require.main.require('./common/auth/tokenManager');
const InvalidTokenError = require.main.require('./common/auth/errors/InvalidTokenError');
const log = require.main.require('./common/logger');

const ObjectID = require('mongodb').ObjectID;

class UserToken extends TokenManager {
  constructor(User, passedConfig) {
    super(passedConfig.auth.token);

    this.User = User;
  }

  signAccessToken(user) {
    // TODO: implement better check that no passwords stored
    const tokenData = {};
    for (const prop in user) {
      if (prop !== 'password' && prop !== 'hashedPassword'
      && prop !== 'iat' && prop !== 'exp') {
        tokenData[prop] = user[prop];
      }
    }

    return super.signAccessToken(tokenData);
  }

  sendToken(tokenHash, res) {
    const token = {};
    token[this.tokenName] = tokenHash;
    res.json(token);

    return res;
  }

  /* @param tokenData : {id_token:<String>,refresh_token:<String>}
    @param refreshTokenPayload optional : refresh existing refresh token
  */
  sendTokenWithRefresh(idTokenHash, refreshTokenHash, res) {
    const idToken = {};
    idToken[this.tokenName] = idTokenHash;

    this.setRefreshTokenCookie(refreshTokenHash, res).json(idToken);

    return res;
  }

  setRefreshTokenCookie(refreshTokenHash, res) {
    res.cookie(this.refreshTokenName, refreshTokenHash, {
        maxAge: this.refreshExpiration, // * 1000,
        httpOnly: true,
        secure: true,
        //TODO: set this to true: secure: 
      }
    )

    return res;
  }

  clearRefreshTokenFromHeader(res) {
    res.clearCookie(this.refreshTokenName);
    return res; 
  }

  //Validates and refreshes the idToken
  //The refreshToken is just checked for integrity, is never rehydrated
  //At the moment, middleware that relies on this does not expect the refresh
  //Token to be modified
  refreshTokens(...args) {
    const cb = args.pop();
    const idToken = args[0];
    const refreshToken = args[1];

    //Currently only applies to id tokens; refresh tokens never expire, are instead revoked
    const rehydrate = args[2] || false;

    if(!idToken) {
      cb(new InvalidTokenError('missing_token', {message: 'invalid tokens'}, 401));
      return;
    }

    // ignore expiration to allow decode token to be passed to callback
    // this means no error raised for expiration
    this.verifyAccessToken(idToken, {ignoreExpiration: 1}, (vErr, dIdToken) => {
      if (vErr) {
        //Keep error messages opaque, to reduce information passed to attackers
        cb(new InvalidTokenError('invalid_token', {message: 'invalid tokens'}, 401));
        return;
      }

      // If not expired, we have nothing to do
      if(!this.isExpired(dIdToken)) {
        cb(null, idToken, refreshToken, dIdToken)
        return;
      }

      // If no refresh token supplied, can't proceed to refresh access token
      if(!refreshToken) {
        cb(new InvalidTokenError('missing_refresh_token', {message: 'invalid tokens'}, 401));
        return;
      }

      // Look up user by expired token's id; each user has single token
      this.User.collection.findOne({
        _id: ObjectID(dIdToken.id), refreshToken: refreshToken
      }, {
        refreshToken: 1,
      }, (err, user) => {
        if(err || !user) {
          cb(new InvalidTokenError('mismatched_refresh_token', {message: 'invalid tokens'}, 401));
          return;
        }

        // Validate the refresh token, and rehydrate the access token if valid
        // For now we just check integrity of refresh token, and don't care
        // about its expiration
        // TODO: check the security of this strategy
        // Note: no error will be raised for expiration, due to ignoreExpiration
        this.verifyRefreshToken(user.refreshToken, {ignoreExpiration: 1}, (err) => {
          // let hydratedRefreshToken = this.signRefreshToken();
          if (err) {
            this.User.collection.update({_id: ObjectID(dIdToken.id)}, {
              $unset: {refreshToken: ''}
            }, (uErr) => {
              // TODO: this could be problematic, since refresh token could get out of sync
              // with browser's cookie
              if(uErr) {
                log.error(uErr);
                cb(new InvalidTokenError('server_error', {message: uErr.message}, 500));

                return;
              }

              // TODO: Can err here be a server error?
              cb(new InvalidTokenError('invalid_refresh_token', {message: 'invalid tokens'}, 401));
            });
          } else {
            // Rehydrate the id/access token if wanted, and send off to user
            cb(null, rehydrate ? this.signAccessToken(dIdToken) : refreshToken, refreshToken, dIdToken);
          }
        });
      });
    });
  }
}

exports = module.exports = UserToken;
