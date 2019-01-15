// Modified from express-jwt, adapted for Polka
const jwt = require('jsonwebtoken');
const async = require('async');
const { set } = require('lodash');

const InvalidTokenError = require.main.require(
  './common/auth/errors/InvalidTokenError'
);

const DEFAULT_REVOKED_FUNCTION = function(_, __, cb) {
  return cb(null, false);
};

const isFunction = object =>
  Object.prototype.toString.call(object) === '[object Function]';

const wrapStaticSecretInCallback = secret => (_, __, cb) => cb(null, secret);

const defaultTokenFn = req => {
  if (!(req.headers && req.headers.authorization)) {
    return null;
  }

  const parts = req.headers.authorization.split(' ');

  if (parts.length === 2) {
    if (/^Bearer$/i.test(parts[0])) {
      return parts[1];
    }

    return null;
  }

  // TODO: Don't give any indication to client that we know
  // that they gave us a bad header
  // unless credentials are required?
  // next(new InvalidTokenError());
  return null;
};

// Different from exprss-jwt: configure once, use multiple times
module.exports = options => {
  if (!options || !options.secret) throw new Error('secret should be set');

  let secretCallback = options.secret;

  if (!isFunction(secretCallback)) {
    secretCallback = wrapStaticSecretInCallback(secretCallback);
  }

  const arity = secretCallback.length;

  const isRevokedCallback = options.isRevoked || DEFAULT_REVOKED_FUNCTION;

  const _requestProperty =
    options.userProperty || options.requestProperty || 'user';
  const _resultProperty = options.resultProperty;
  const credentialsRequired =
    typeof options.credentialsRequired === 'undefined'
      ? true
      : options.credentialsRequired;

  const tokenFn = options.getToken ? options.getToken : defaultTokenFn;

  // Slow but called only once
  if (!isFunction(tokenFn)) {
    throw new Error('getToken must be a function');
  }

  return (req, res, next) => {
    const token = tokenFn(req);

    if (!token) {
      if (credentialsRequired) {
        next(new InvalidTokenError());
      } else {
        next();
      }

      return;
    }

    try {
      const dtoken = jwt.decode(token, { complete: true });

      if (!dtoken) {
        // TODO: Maybe just throw regardless of credentialsRequired?
        if (credentialsRequired) {
          next(new InvalidTokenError());
        } else {
          next();
        }
      }

      async.waterfall(
        [
          function getSecret(callback) {
            if (arity === 4) {
              secretCallback(req, dtoken.header, dtoken.payload, callback);
            } else {
              // arity == 3
              secretCallback(req, dtoken.payload, callback);
            }
          },
          function verifyToken(secret, callback) {
            jwt.verify(token, secret, options, (err, decoded) => {
              if (err) {
                callback(new InvalidTokenError());
              } else {
                callback(null, decoded);
              }
            });
          },
          function checkRevoked(decoded, callback) {
            isRevokedCallback(req, dtoken.payload, (err, revoked) => {
              if (err) {
                callback(err);
              } else if (revoked) {
                callback(new InvalidTokenError());
              } else {
                callback(null, decoded);
              }
            });
          }
        ],
        (err, result) => {
          if (err) {
            next(err);
            return;
          }

          if (_resultProperty) {
            set(res, _resultProperty, result);
          } else {
            set(req, _requestProperty, result);
          }

          next();
        }
      );
    } catch (err) {
      next(new InvalidTokenError());
    }
  };
};
