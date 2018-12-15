function InvalidTokenError(message) {
  Error.call(this, message);
  Error.captureStackTrace(this, this.constructor);
  this.name = 'InvalidTokenError';
  this.message = message;
  this.code = 401;
  this.status = 401;
}

InvalidTokenError.prototype = Object.create(Error.prototype);
InvalidTokenError.prototype.constructor = InvalidTokenError;

module.exports = InvalidTokenError;
