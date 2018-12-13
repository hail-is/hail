const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;

exports.setup = function setupLocalStrategy(User, config, tokenManager) {
  passport.use(new LocalStrategy({
    usernameField: 'email',
    passwordField: 'password',// this is the virtual field on the model
  },
  function passportCb(email, password, cb) {
    User.findOne({
      email: email.toLowerCase(), // TODO: implement in the model itself;
    },
    function findCb(err, user) {
      if (err) {
        return cb(err, false);
      }

      // TODO: is this redundant with unique:true restriction in Mongoose?
      if (!user || !user.authenticate(password) ) {
        if(user && user.authenticateSha1(password)) {
          // update hashedPassword via Mongoose setter to use latest schema
          user.password = password;
        } else {
          // don't give attackers granular info
          return cb(null, false);
        }
        
      }

      user.lastLogin = Date.now();

      const token = tokenManager.signRefreshToken();
      user.refreshToken = token;

      user.save(cb); 
    });
  }) );
};
