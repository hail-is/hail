function InvalidTokenError(code, error, status) {
  Error.call(this, error.message);
  Error.captureStackTrace(this, this.constructor);
  this.name = 'InvalidTokenError';
  this.message = error.message;
  this.code = code;
  this.status = status || 401;
  this.inner = error;
}

InvalidTokenError.prototype = Object.create(Error.prototype);
InvalidTokenError.prototype.constructor = InvalidTokenError;

module.exports = InvalidTokenError;
