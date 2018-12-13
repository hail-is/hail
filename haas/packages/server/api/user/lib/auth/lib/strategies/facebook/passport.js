const passport = require('passport');
const FacebookStrategy = require('passport-facebook').Strategy;

exports.setup = function setupFacebookStrategy(User, config) {
  console.log('in facebook we have config:', config);
  passport.use(new FacebookStrategy({
    clientID: config.clientID,
    clientSecret: config.clientSecret,
    callbackURL: config.callbackURL,
  },
  function passportCb(accessToken, refreshToken, profile, done) {
    User.findOne({
      'facebook.id': profile.id,
    },
    function findCb(err, user) {
      let thisUser = user;
      if (err) {
        return done(err);
      }
      if (!thisUser) {
        thisUser = new User({
          name: profile.displayName,
          email: profile.emails[0].value,
          username: profile.username,
          provider: 'facebook',
          facebook: profile._json,
        });
        thisUser.save(function saveCb(saveErr) {
          if (saveErr) done(saveErr);
          return done(saveErr, thisUser);
        });
      } else {
        return done(err, thisUser);
      }
    });
  }) );
};
