const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth').OAuth2Strategy;

exports.setup = function setupGoogleStrategy(User, config) {
  passport.use(new GoogleStrategy({
    clientID: config.clientID,
    clientSecret: config.clientSecret,
    callbackURL: config.callbackURL,
  },
  function passportCb(accessToken, refreshToken, profile, done) {
    User.findOne({
      'google.id': profile.id,
    }, function findCb(err, user) {
      if (!user) {
        const registerUser = new User({
          name: profile.displayName,
          email: profile.emails[0].value,
          role: 'user',
          username: profile.username,
          provider: 'google',
          google: profile._json,
        });
        registerUser.save(function saveCb(saveErr) {
          if (saveErr) done(saveErr);
          return done(saveErr, registerUser);
        });
      } else {
        return done(err, user);
      }
    });
  }) );
};
