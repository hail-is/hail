const log = require.main.require('./common/logger');
const crypto = require('crypto');

module.exports.set = function set(Schema, options) {
  Schema.plugin(resetPlugin, options);
};

function resetPlugin(schema, options) {
  const expires = 108000 || options.expires; // in ms, default 30m

  schema.add({
    reset: {
      token: String,
      expires: Date,
    },
  });

  schema
  .pre('save', function preSave(next, done) {
    // any time anything is changed, if we have a reset token, erase it.
    if (!this._reset) {
      this.reset = undefined;
    } else if (this.reset && this.reset.token) { 
      if (!this.reset.expires || this.reset.expires > Date.now() + expires) {
        this.invalidate('reset', 'token must have proper expiration');
        return done(new Error('Reset token set w/o proper expiration! Erasing') );
      }
    }
    next();
  });

  schema.methods.makeResetToken = function() {
    this.reset.token = this.encryptPassword(
      crypto.randomBytes(512), this.makeSalt()
    );
    this.reset.expires = Date.now() + expires;
    this._reset = true;
    return this.reset.token;
  };

  schema.statics.validateResetToken = function(token, cb) {
    this.findOne({
      'reset.token': token, 
      'reset.expires': {$lt: Date.now() + expires} 
    }, function(err, user) {
      if (err) {
        log.error(err);
        return cb(err, false);
      }
      return cb(null, user);
    });
  };

  schema.statics.resetPassword = function(token, newPass, cb) {
    this.validateResetToken(token, function(err, user) {
      if (err) { return cb(err); }
      if (!user) { return cb(null, null); }
      user.password = newPass;
      user.reset = undefined;
      user.save(cb);
    });
  };
}
