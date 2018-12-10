const mailer = require('./mail');
const log = require.main.require('./common/logger');

let UserModel;

exports.setup = function setupGuestStrategy(User) {
  UserModel = User;
};

exports.resetPassword = function resetPassword(req, cb) {
  const tCb = cb || function() {};
  const password = req.body.password || req.query.password;
  const token = decodeURIComponent(req.body.token || req.query.token);
  if (!(password && token) ) {
    log.warn('Password reset attempted without password or token', password, token);
    return tCb(null, null);
  }
  log.debug('the token we received in resetPassword is', token);
  UserModel.resetPassword(token, password, tCb);
};

exports.sendResetLink = function sendResetLink(req, cb) {
  const tCb = cb || function() {};
  const email = decodeURIComponent(req.body.email || req.query.email);

  if (!email) {
    return cb(null, null);
  }

  log.debug('mail is', email);

  UserModel.findOne({email: email}, function findOneCb(err, user) {
    if (err) { return tCb(err); }
    if (!user) { return tCb(null, false); }
    const token = user.makeResetToken();
    user.save(function saveCb() {
      // token persisted
      log.debug('token is', token);
      log.debug('user is', user);
      mailer.sendResetEmail(email, token, tCb);
    });
  });
};


