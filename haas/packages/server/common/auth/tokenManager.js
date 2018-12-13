// TODO: Finish implementing refresh token logic
const jwksRsa = require('jwks-rsa');

class TokenManager {
  constructor(config) {
    this.tokenName = config.tokenName || 'access_token';
    this.attachProperty = config.attachProperty || 'user';

    this.secret = jwksRsa.expressJwtSecret({
      cache: true,
      rateLimit: true,
      jwksRequestsPerMinute: 5,
      jwksUri: process.env.AUTH0_WEB_KEY_SET_URL
    });
  }

  getAccessTokenFromRequest(req) {
    const maybeToken =
      req.query[this.tokenName] ||
      req.params[this.tokenName] ||
      req.body[this.tokenName];

    if (maybeToken) {
      return maybeToken;
    }

    if (req.headers && req.headers.authorization) {
      const parts = req.headers.authorization.split(' ');
      if (parts.length === 2) {
        const scheme = parts[0];
        const credentials = parts[1];

        if (/^Bearer$/i.test(scheme)) {
          return credentials;
        }
      }
    }

    return null;
  }

  // isExpired(decodedToken) {
  //   return Math.floor(Date.now() / 1000) >= decodedToken.exp;
  // }
}

module.exports = TokenManager;
