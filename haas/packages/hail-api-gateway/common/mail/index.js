const nodemailer = require('nodemailer');
const smtpPool = require('nodemailer-smtp-pool');
const xoauth2 = require('xoauth2');

if (
  !(
    process.env.googleMailerClientId &&
    process.env.googleMailerClientSecret &&
    process.env.refreshToken &&
    process.env.SERVER_ADDRESS
  )
) {
  throw new Error(`Require process.env to have 
    oauth2 googleMailerClientId, googleMailerClientSecret, refreshToken, and SERVER_ADDRESS prop/valuees in
    sendmail`);
}
// TODO: base on process.ENV
const generator = xoauth2.createXOAuth2Generator({
  user: process.env.googleMailerUser, // ex: user@gmail.com
  clientId: process.env.googleMailerClientId,
  clientSecret: process.env.googleMailerClientSecret,
  refreshToken: process.env.refreshToken
});

const smtpTransport = nodemailer.createTransport(
  smtpPool({
    service: 'gmail',
    auth: {
      xoauth2: generator
    },
    maxConnections: 10,
    rateLimit: 100
  })
);

function mailApi(userEmailAddress, subject, userMessage, pFrom, cb) {
  const tCb = cb || function() {};

  const mailComp = {
    // sender info
    from: `${pFrom || 'Hail Admin'} <akotlar@broadinstitute.org>`,

    // Comma separated list of recipients
    to: userEmailAddress,

    // Subject of the message
    subject: subject, //

    // HTML body
    html: userMessage
  };

  smtpTransport.sendMail(mailComp, function(error, info) {
    // if you don't want to use this transport object anymore, uncomment following line
    // transport.close(); // close the connection pool
    return tCb(error, info);
  });
}

module.exports.send = mailApi;
