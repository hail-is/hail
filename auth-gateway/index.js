require("dotenv").config();

const polka = require("polka");
const cookie = require("cookie");
const jwksRsa = require("jwks-rsa");
const jwt = require("jsonwebtoken");

const { AUTH0_DOMAIN, AUTH0_AUDIENCE } = process.env;
const AUTH0_ISSUER = `https://${AUTH0_DOMAIN}/`;

// Grabs public key, caches it, from Auth0's servers
// asymmetric signature used to ensure that not only the jwt was unmodified
// since signed, but that the issuer was Auth0 (since they hold the private key)
const jwksFn = jwksRsa({
  cache: true,
  rateLimit: true,
  jwksRequestsPerMinute: 5,
  jwksUri: `https://${AUTH0_DOMAIN}/.well-known/jwks.json`
});

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

const verifyToken = token =>
  new Promise((resolve, reject) => {
    jwt.verify(
      token,
      getSecretKey,
      {
        issuer: AUTH0_ISSUER,
        audience: AUTH0_AUDIENCE,
        clockTolerance: 2, // seconds to deal with clock skew
        algorithms: ["RS256"]
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

const bearerPrefix = "Bearer ";
const bearerPrefixLen = bearerPrefix.length;

const cookiePrefix = "access_token=";
const cookieOffset = cookiePrefix.length;

const getAuthToken = req => {
  const token = req.headers.cookie
    ? cookie.parse(req.headers.cookie)["access_token"]
    : null;

  if (token === null || token === "") {
    return null;
  }

  return token;
};

const PORT = 8000;
polka()
  .get("/verify", (req, res) => {
    const token = cookie.parse(req.headers.cookie)["access_token"];

    if (token === null) {
      unauthorized(res);
      return;
    }

    verifyToken(token)
      .then(user => {
        res.setHeader("User", user.sub);
        res.setHeader("Scope", user.scope);

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

module.exports = {
  getAuthToken
};
