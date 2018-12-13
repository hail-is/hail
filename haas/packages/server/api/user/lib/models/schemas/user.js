const crypto = require('crypto');

const { Schema } = require.main.require('./common/database');

const assert = require('assert');

const accountName = process.env.APP_NAME.toLowerCase();

console.info('account name is', accountName);

module.exports = function userSchemaFactory() {
  // lowest level role is always the one with least priveleges
  // lowest level role doesn't persist
  const userSchema = new Schema(
    {
      name: { type: String, trim: true },
      // username: { type: String, trim: true, unique: true},
      email: { type: String, lowercase: true, trim: true, unique: true },
      hashedPassword: String,
      salt: String,
      lastLogin: Date,
      seenNotifications: Object,
      options: {
        autoUploadToS3: { type: Boolean, default: false }
      },
      cloud: {
        s3: {
          credentials: {
            accessID: String,
            secret: String
          }
        }
      },
      facebook: Object,
      twitter: Object,
      google: Object,
      github: Object,
      accounts: { type: Array, default: [accountName] },
      refreshToken: String,
      hints: Object
    },
    { collection: 'Users' }
  );

  userSchema.set('toJSON', {
    transform: (doc, ret) => {
      // Do not expose accessID, secret
      if (
        ret.cloud &&
        ret.cloud.s3 &&
        ret.cloud.s3.credentials &&
        ret.cloud.s3.credentials.secret
      ) {
        ret.cloud.s3.credentials.secret = 'hidden';
      }

      delete ret.salt;
      delete ret.hashedPassword;
      delete ret.__v;
      return ret;
    }
  });

  /**
   * Virtuals
   */
  userSchema
    .virtual('password')
    .set(function setPassword(password) {
      // console.info('thi in password virtual',this);
      this._password = password; // won't get saved to db unless in schema
      this.salt = this.makeSalt();
      this.hashedPassword = this.encryptPassword(password);
    })
    .get(function getPassword() {
      return this._password;
    });

  // TODO, probably change this name, and the above, these aren't exaclty tokens
  // these are token objects
  // TODO: use https://github.com/T-PWK/flake-idgen

  /**
   * Validations
   */
  userSchema
    .path('email')
    .validate(email => email.length > 0, 'Email cannot be blank');

  // Validate empty password
  userSchema
    .path('hashedPassword')
    .validate(
      hashedPassword => hashedPassword.length > 0,
      'Password cannot be blank'
    );

  // Validate email is not taken
  // TODO: is this needed if defining unique : true on email property
  userSchema.path('email').validate(function validateEmail(value) {
    const self = this;

    return (
      this.constructor
        .findOne({ email: value })
        // Allow user to either be new, or allow email address to be updated
        .then(user => assert.ok(!user || self.id === user.id))
    );
  }, 'The specified email address is already in use.');

  /**
   * Pre-save hook
   */
  userSchema.pre('save', function preSave(next) {
    if (!this.isNew) return next();

    if (!(this.hashedPassword && this.hashedPassword.length)) {
      // authTypes.indexOf(this.provider) === -1) {
      return next(new Error('Invalid password'));
    }

    return next();
  });

  /**
   * Methods
   */
  userSchema.methods = {
    // return { authenticate, makeSalt, encryptPassword };

    /**
     * Authenticate - check if the passwords are the same
     *
     * @param {String} plainText
     * @return {Boolean}
     * @api public
     */
    authenticate: function authenticate(plainText) {
      return this.encryptPassword(plainText) === this.hashedPassword;
    },

    // we want to gracefully upgrade users to SHA512; allow them to use the
    // previous schema for their first login
    authenticateSha1: function authenticate(plainText) {
      return this.encryptPasswordSha1(plainText) === this.hashedPassword;
    },

    /**
     * Make salt
     *
     * @return {String}
     * @api public
     */
    makeSalt: function makeSalt() {
      return crypto.randomBytes(32).toString('base64');
    },

    /**
     * Encrypt password
     *
     * @param {String} password
     * @return {String}
     * @api public
     */
    // 64 bytes here is proper; native output of sha512
    // 41 days with 39 bits of entropy on 4 gpu system
    // https://support.1password.com/pbkdf2/
    // https://blog.agilebits.com/2014/03/10/crackers-report-great-news-for-1password-4/
    // setting this somewhat conservatively; in general the number of iterations
    // not nearly as effective as setting a password with more entropy
    // to the point where adding 1 bit of password entropy is more valuable than
    // 10x iterations from 30k to 300k
    encryptPassword: function encryptPassword(password, altSalt) {
      let salt = altSalt || this.salt;
      if (!password || !salt) return '';
      salt = Buffer.from(salt, 'base64');
      return crypto
        .pbkdf2Sync(password, salt, 100000, 64, 'sha512')
        .toString('base64');
    },

    // 64 byte size is improper; should be 20 (the digest size of sha1)
    // longer lengths take more time to calculate, but attackers will use just the
    // first N bytes, where N is the length of the hash output of the sha (or other) algorithm
    // When requesting more bytes, pbkdf2 will concatenate the results of the first native hash length (bytes)
    // with the results of each following output; the attacker will need only crack the
    // first part
    encryptPasswordSha1: function encryptPassword(password, altSalt) {
      let salt = altSalt || this.salt;
      if (!password || !salt) return '';
      salt = Buffer.from(salt, 'base64');
      return crypto
        .pbkdf2Sync(password, salt, 10000, 64, 'sha1')
        .toString('base64');
    }
  };

  // logger.debug('user schema instance methods', userSchema.methods);

  return userSchema;
};
