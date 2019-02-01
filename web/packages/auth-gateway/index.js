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

const PORT = 8000;
// Polka is *mostly* express compatible, but let's not use unnecessary libraries
// better to understand the underlying security issues
// Best practices state that we should check the exact form of the auth header
// and if accepting from more than 1 place (say body or get query)
// make sure the token exists in only one place
// We have no need for this. Accept only from the header

// Also, I prefer not using middleware; call the functions needed inside the route
const bearerPrefix = 'Bearer ';
// naturally 1 past; so useful as end, and as the index to start reading token from
const bearerPrefixLen = bearerPrefix.length;
const getAuthToken = req => {
  // This is set from an "Authorization" header
  if (req.headers.authorization) {
    if (req.headers.authorization.substr(0, bearerPrefixLen) !== bearerPrefix) {
      return null;
    }

    const token = req.headers.authorization.substr(bearerPrefixLen);

    if (token.trim() === '') {
      return null;
    }

    return token;
  }

  if (req.query.access_token) {
    return req.query.access_token;
  }

  return null;
};

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
