const mailer = require.main.require('./common/mail');
const log = require.main.require('./common/logger');

if (!process.env.SERVER_ADDRESS) {
  throw new Error(`Require process.env to have SERVER_ADDRESS prop/values`);
}

const name = process.env.APP_NAME;

/* Note that the url to reset must be set to /user/reset?token= in SPA */
module.exports.sendResetEmail =
function sendResetEmail(email, token, cb) {
  log.debug('struff we\'re trying to send', token);
  const tToken = encodeURIComponent(token);
  const message =
    `<p>${name} received a reset link request, for the account attached to 
        this email address.
    </p> 
    <p>If you did not request this link
    please email the <a href='mailto:seqadm1@gmail.com'>${name} Admin</a> immediately</p>
    <p>If you wish to reset your password, please click on:</p>
    <p>
      <a href="${process.env.SERVER_ADDRESS}/user/reset?token=${tToken}">
        Reset Link
      </a>
    </p>`;
  mailer.send(email, name + ' Password Reset', message, name + ' Admin', cb);
};

