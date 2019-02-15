require('dotenv').config();

const polka = require('polka');
const jwksRsa = require('jwks-rsa');
const jwt = require('jsonwebtoken');

const { AUTH0_DOMAIN, AUTH0_AUDIENCE } = process.env;
const AUTH0_ISSUER = `https://${AUTH0_DOMAIN}/`;

// Grabs public key, caches it, from Auth0's servers
// asymeetric signature used to ensure that not only the jwt was unmodified
// since signed, but that the issuer was Auth0 (since they hold the private key)
const jwksFn = jwksRsa({
  cache: true,
  rateLimit: true,
  jwksRequestsPerMinute: 5,
  jwksUri: `https://${AUTH0_DOMAIN}/.well-known/jwks.json`
});

// TODO: Validate audience, issuer

// Ref: https://github.com/auth0/node-jwks-rsa/blob/master/src/integrations/express.js
function getSecretKey(header, cb) {
  jwksFn.getSigningKey(header.kid, (err, key) => {
    if (err) {
      cb(err, null);
      return;
    }

    cb(null, key.publicKey || key.rsaPublicKey);
  });
}

// Verifies signature, as well as the issuer and audience, to guarantee
// that the token was truly for our resource server
const verifyToken = token =>
  new Promise((resolve, reject) => {
    jwt.verify(
      token,
      getSecretKey,
      {
        issuer: AUTH0_ISSUER,
        audience: AUTH0_AUDIENCE,
        clockTolerance: 2, // seconds to deal with clock skew
        algorithms: ['RS256']
      },
      (err, dToken) => {
        if (err) {
          console.error(err);
          return reject(err);
        }

        return resolve(dToken);
      }
    );
  });

// Polka is *mostly* express compatible, but let's not use unnecessary libraries
// better to understand the underlying security issues
// Best practices state that we should check the exact form of the auth header
// and if accepting from more than 1 place (say body or get query)
// make sure the token exists in only one place
// We have no need for this. Accept only from the header

// Also, I prefer not using middleware; call the functions needed inside the route
const bearerPrefix = 'Bearer ';
const bearerPrefixLen = bearerPrefix.length;

const cookiePrefix = 'access_token=';
const cookieOffset = cookiePrefix.length;

const getAuthToken = req => {
  let whereFoundCount = 0;

  let bearerIdx;
  let cookieIdx;

  if (req.headers.cookie) {
    cookieIdx = req.headers.cookie.indexOf(cookiePrefix);

    if (cookieIdx > -1) {
      whereFoundCount += 1;
    }
  }

  if (req.query.access_token) {
    whereFoundCount += 1;
  }

  if (req.headers.authorization) {
    bearerIdx = req.headers.authorization.indexOf(bearerPrefix);

    if (bearerIdx > -1) {
      whereFoundCount += 1;
    }
  }

  // oauth2 spec specifies token should be presented in only one location
  if (whereFoundCount === 0) {
    return null;
  }

  if (whereFoundCount > 1) {
    console.warn('User specified > 1 access tokens', req.headers.origin);
    return null;
  }

  // The method we are least likely to use; disable for now
  // as this means access tokens are even easier to abuse
  // if (req.query.access_token) {
  //   return req.query.access_token;
  // }

  let token;

  if (bearerIdx > -1) {
    token = req.headers.authorization.substr(bearerPrefixLen);
  } else if (cookieIdx > -1) {
    token = req.headers.cookie.substr(cookieIdx + cookieOffset);
  }

  if (!token) {
    return null;
  }

  const spaceIdx = token.indexOf(' ');

  // We have a single authorization value, and it is the Bearer token
  if (spaceIdx === -1) {
    return token;
  }

  // When multiple authorization headers, they are delimited by a space and a ,
  // https://stackoverflow.com/questions/29282578/multiple-http-authorization-headers
  // Similarly, when multiple cookies present, separated by value1, cookie2=value2
  // or likewise by semicolon
  return token.substr(0, spaceIdx - 1);
};

const PORT = 8000;
polka()
  .get('/verify', (req, res) => {
    const token = getAuthToken(req);

    if (!token) {
      unauthorized(res);
      return;
    }

    verifyToken(token)
      .then(user => {
        res.setHeader('User', user.sub);
        res.setHeader('Scope', user.scope);

        res.end();
      })
      .catch(e => {
        console.error(e);
        unauthorized(res);
      });
  })
  .listen(PORT, err => {
    if (err) {
      throw err;
    }

    console.info(`Auth server running on port ${PORT}`);
  });

function unauthorized(res) {
  res.statusCode = 401;
  res.end();
}
