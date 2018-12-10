// const log = require.main.require('./common/logger');
const AWS = require('aws-sdk');
const { createError } = require('apollo-errors');

// TODO: Retrieve from process.env (yaml)
const s3Bucket = process.env.S3_BUCKET;
const s3Exception = createError('S3Exception', {
  message: `Couldn't retrieve data from your the ${Bucket} bucket`
});

const {
  errors: { AuthError }
} = require.main.require('./common/graphql');

const s3 = new AWS.S3({
  accessKeyId: process.env.S3_ACCESS_KEY,
  secretAccessKey: process.env.S3_SECRET_KEY
});

// https://stackoverflow.com/questions/45625548/how-to-upload-an-image-to-aws-s3-using-graphql

const analysisResolver = {
  Query: {
    // job: (parent, args, { id }) => jobs[id],
    // allJobs: () => jobs,
    s3File: async (parent, { url }, { user }) => {
      if (!user) {
        throw new AuthError();
      }

      console.info('called s3File', url);
      try {
        const data = await s3
          .getObject({
            Bucket: s3Bucket,
            Key: url
          })
          .promise();

        const str = data.Body.toString('utf-8'); // Use the encoding necessary
        // console.info(data, JSON.stringify(d3.tsvParse(str)));

        return {
          contents: str
        };
      } catch (error) {
        console.info('error', error);
        throw new s3Exception();
      }
    },
    allS3Objects: async (parent, vars, { user }) => {
      if (!user) {
        throw new AuthError();
      }

      try {
        const response = await s3
          .listObjects({
            Bucket: s3Bucket
          })
          .promise();

        if (response.Contents.length === 0) {
          return [];
        }

        return response.Contents.map(val => {
          return {
            name: val.Key,
            lastModified: val.LastModified,
            ownerName: val.Owner.DisplayName
          };
        });
      } catch (error) {
        // log.error(error);
        throw new s3Exception();
      }
    }
    // _allPostsMeta: () => {
    //   return { count: posts.length };
    // }
  }
  // User: {
  //   username: user => `${user.firstname} ${user.lastname}`
  // },
  // Message: {
  //   user: (parent, args, context) => {
  //     return users[parent.userId];
  //   }
  // }
};

module.exports = analysisResolver;
