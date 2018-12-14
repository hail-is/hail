const log = require.main.require('./common/logger');
const aws4 = require('aws4');
const crypto = require('crypto');
const AWS = require('aws-sdk');

/*
  @public
*/
// TODO: improve error handling
function validationError(res, err) {
  return res.status(422).json(err || null);
}

exports = module.exports = function awsController(User) {
  return { listS3Buckets, listS3bucketContents, createS3signature };
  /**
   * Generate an AWS signature
   * restriction: 'admin'
   */
  function listS3Buckets(req, res) {
    if (!req.user.id) {
      if (!req.user.id) {
        return res.sendStatus(401);
      }
    }

    User.findById(req.user.id, { cloud: 1 }, (err, user) => {
      if (err) {
        $log.error(err);
        return res.sendStatus(500);
      }

      if (!user) {
        return res.send(404).send('User not found');
      }

      const accessID = user.cloud.s3.credentials.accessID;
      const secret = user.cloud.s3.credentials.secret;

      if (!accessID && secret) {
        return res.send(404).send('Missing s3 credentials');
      }

      const s3 = new AWS.S3({
        accessKeyId: accessID,
        secretAccessKey: secret
      });

      s3.listBuckets((err, data) => {
        if (err) {
          log.error(err);
          return res.status(404).send(err.code);
        } else {
          return res.json(data);
        }
      });
    });
  }

  function listPublicBucketContents(req, res) {
    const s3 = new AWS.S3();

    s3.listObjects(
      {
        Bucket: '/eqant-paper'
      },
      (err, data) => {
        if (err) {
          log.error(err);
          return res.status(404).send(err.code);
        } else {
          console.info('bucket data', data);
          return res.json(data);
        }
      }
    );
  }

  function listS3bucketContents(req, res) {
    if (!req.user.id) {
      if (!req.user.id) {
        return res.sendStatus(401);
      }
    }

    User.findById(req.user.id, { cloud: 1 }, (err, user) => {
      if (err) {
        $log.error(err);
        return res.sendStatus(500);
      }

      if (!user) {
        return res.send(404).send('User not found');
      }

      const accessID = user.cloud.s3.credentials.accessID;
      const secret = user.cloud.s3.credentials.secret;

      if (!accessID && secret) {
        return res.send(404).send('Missing s3 credentials');
      }

      const s3 = new AWS.S3({
        accessKeyId: accessID,
        secretAccessKey: secret
      });

      s3.listObjects(
        {
          Bucket: req.params.bucketName
        },
        (err, data) => {
          if (err) {
            log.error(err);
            return res.status(404).send(err.code);
          } else {
            console.info('bucket data', data);
            return res.json(data);
          }
        }
      );
    });
  }

  function createS3signature(req, res) {
    if (!req.user.id) {
      return res.sendStatus(401);
    }

    // TODO: should we save people's secrets?
    const awsSecretKey = req.body.awsSecretKey;
    const awsAccessId = req.body.awsAccessId;
    const bucketName = req.body.bucketName;
    const acl = 'private' || req.body.acl;

    const region = 'us-east-1' || req.body.region;

    console.info('req.body is', req.body);
    //TODO: allow user to set expiration
    //Thanks to https://github.com/danialfarid/ng-file-upload/issues/1128#issuecomment-216253393
    const date =
      new Date()
        .toISOString()
        .replace(/[\.\-:]/gi, '')
        .substr(0, 15) + 'Z';
    const dateNowRaw = date.substr(0, date.indexOf('T'));

    const expirationDate = new Date();
    expirationDate.setHours(expirationDate.getHours() + 1);
    const expiration = expirationDate.toISOString();

    const credentials = awsAccessId + dateNowRaw + `/${region}/s3/aws4_request`;

    console.info('credential is', credentials);
    const policy = {
      expiration: expiration,
      conditions: [
        { bucket: bucketName },
        { acl: acl },
        ['starts-with', '$key', ''],
        ['starts-with', '$Content-Type', ''],
        { 'x-amz-credential': credentials },
        { 'x-amz-algorithm': 'AWS4-HMAC-SHA256' },
        { 'x-amz-date': date }
      ]
    };

    const opts = {
      service: 's3',
      region: region
    };

    aws4.sign(opts, {
      accessKeyId: awsAccessId,
      secretAccessKey: awsSecretKey
    });

    console.info('after signing, opts', opts);

    const base64Policy = Buffer.from(JSON.stringify(policy), 'utf-8').toString(
      'base64'
    );

    const dateKey = crypto
      .createHmac('sha256', 'AWS4' + awsSecretKey)
      .update(dateNowRaw)
      .digest();
    const dateRegionKey = crypto
      .createHmac('sha256', dateKey)
      .update(region)
      .digest();
    const dateRegionServiceKey = crypto
      .createHmac('sha256', dateRegionKey)
      .update('s3')
      .digest();
    const signingKey = crypto
      .createHmac('sha256', dateRegionServiceKey)
      .update('aws4_request')
      .digest();

    const signature = crypto
      .createHmac('sha256', signingKey)
      .update(base64Policy)
      .digest('hex');

    res.status(200).json({
      signature: signature,
      policy: base64Policy,
      date: date,
      credentials: credentials,
      expiration: expiration
    });
    // userModel.findById(req.user.id, (err, user) => {
    //   if (err) {
    //     return res.status(500).send(err);
    //   }

    //   if (!user) {
    //     return res.status(401).send(err);
    //   }

    //   if(!user.awsSignatures) {
    //     user.awsSignatures = {};
    //   }

    //   user.awsSignatures.s3 = signature;

    //   res.status(200).json(users);
    // });
  }
};
