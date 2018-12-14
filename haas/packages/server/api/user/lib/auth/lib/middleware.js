const expressJwt = require('express-jwt');
const fetch = require('isomorphic-unfetch');
const { promisify } = require('util');

const InvalidTokenError = require.main.require(
  './common/auth/errors/InvalidTokenError'
);

const redisClient = require('redis').createClient({
  parser: 'hiredis'
});

const getAsync = promisify(redisClient.get).bind(redisClient);
const setAsync = promisify(redisClient.set).bind(redisClient);

const tokenUrl = process.env.AUTH0_MANAGEMENT_API_TOKEN_URL;
const managementUrl = process.env.AUTH0_MANAGEMENT_API_URL;

const options = {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    grant_type: 'client_credentials',
    client_id: process.env.AUTH0_MANAGEMENT_API_CLIENT,
    client_secret: process.env.AUTH0_MANAGEMENT_API_SECRET,
    audience: process.env.AUTH0_MANAGEMENT_API_AUDIENCE
  })
};

const auth0oauthTokenPromise = fetch(tokenUrl, options);

const findProviderObject = (userManagementResponse, userID) => {
  // [providerName, id]
  const idParts = userID.split('|');

  const githubObject = userManagementResponse.identities.find(
    // interestingly exact comparison of strings matches,
    // clearly does not treat String(val) as an object of string class
    // since obj1 !== obj2 unless &obj1 === &obj2
    o => o.provider === idParts[0] && String(o.user_id) === idParts[1]
  );

  return githubObject;
};

class AuthMiddleware {
  constructor(User, tokenManager) {
    this.User = User;
    this.tokenMan = tokenManager;

    // sub is what auth0 uses
    this.getUserId = user =>
      // sub is what auth0 uses
      user && (user.id || user._id || user.sub);

    this.verifyTokenPermissiveFn = expressJwt({
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
      credentialsRequired: true
    });

    this.getAuth0ProviderAccessToken = this.getAuth0ProviderAccessToken.bind(
      this
    );

    // elase not guaranteed to respect this
    this.hasRole = this.hasRole.bind(this);
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
  verifyTokenPermissive() {
    return (req, res, next) => this.verifyTokenPermissiveFn(req, res, next);
  }

  async getAuth0ProviderAccessToken(req, res, next) {
    // No need to catch, express handles it
    // https://expressjs.com/en/guide/error-handling.html
    const userID = req.user ? this.getUserId(req.user) : null;

    if (!userID) {
      throw new InvalidTokenError();
    }

    const accessToken = await this.extractAccessToken(userID);

    req.accessToken = accessToken;

    next();
  }

  async extractAccessToken(userID) {
    let accessToken;

    // NOTE: This requires userID to change by auth0provide
    // typically auth0, at least social, connections
    // are in the form provider|id
    accessToken = await getAsync(userID);

    if (!accessToken) {
      const oauthRes = await auth0oauthTokenPromise.then(r => r.json());

      const userManagementResponse = await fetch(`${managementUrl}/${userID}`, {
        headers: {
          Authorization: `Bearer ${oauthRes.access_token}`
        }
      });

      const githubData = findProviderObject(
        userManagementResponse.json(),
        userID
      );

      accessToken = githubData.access_token;

      setAsync(userID, accessToken);
    }

    return accessToken;
  }
}

module.exports = AuthMiddleware;
