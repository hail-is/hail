// const PromiseBird = require('bluebird');
const { promisify } = require('util');
const {
  errors: { AuthError, DataFetchError }
} = require.main.require('./common/graphql');

const s3Bucket = process.env.S3_BUCKET;

exports = module.exports = function JobResolver(Jobs, getUserFn) {
  const jobResolver = {
    Query: {
      job: async (parent, { id }, { user }) => {
        if (!user) {
          throw new AuthError();
        }

        const userID = getUserFn(user);

        try {
          const job = await Jobs.findOne({ _id: id, userID })
            .lean()
            .exec();

          job.id = job._id.toString();

          return job;
        } catch (err) {
          // log.error(err);
          console.error(err);

          throw new DataFetchError();
        }
      },
      jobs: async (parent, { type, visibility }, { user }) => {
        if (!user) {
          throw new AuthError();
        }

        const userID = getUserFn(user);

        const query = { userID };
        let wantShared = false;
        // if (req.params.type === 'incomplete') {
        //   query['submission.state'] = incompleteStateRegex;
        // } else if (
        //   req.params.type === 'complete' ||
        //   req.params.type === 'completed'
        // ) {
        //   query['submission.state'] = completeStateRegex;
        // } else if (req.params.type === 'failed') {
        //   query['submission.state'] = failedStateRegex;
        // } else if (req.params.type === 'deleted') {
        //   query.type = 'deleted';
        // } else if (req.params.type === 'shared') {
        //   wantShared = true;
        // }

        // if (req.params.visibility) {
        //   if (req.params.visibility === 'private') {
        //     query.visibility = 'private';
        //   } else if (req.params.visibility === 'public') {
        //     query.visibility = 'public';
        //   }
        // }

        // if (query.visibility !== 'public') {
        //   if (req.user) {
        //     query.userID = req.user.id;
        //   } else {
        //     return _notAuthorized(res);
        //   }
        // }

        if (wantShared) {
          delete query.userID;

          query[`sharedWith.${userID}`] = {
            $gte: 400
          };
        }

        if (!query.type) {
          query.type = {
            $ne: 'deleted'
          };
        }

        // Fast, but we currently need a better solution to populate the job name
        try {
          const jobs = await Jobs.find(query, {
            results: 0,
            dirs: 0,
            __v: 0,
            'inputQueryConfig.indexName': 0,
            'inputQueryConfig.indexType': 0,
            'search.indexName': 0,
            'search.indexType': 0
          })
            .lean()
            .exec();

          return jobs.map(job => {
            job.id = job._id.toString();
            return job;
          });
        } catch (err) {
          // log.error(err);
          console.error(err);

          throw new DataFetchError();
        }
      }

      // Return the record of files uploaded from your DB or API or filesystem.
    },
    Mutation: {
      async submitJob(parent, { jobConfig }, { user }) {
        if (!user) {
          throw new AuthError();
        }

        const userID = getUserFn(user);

        if (!userID) {
          throw new AuthError();
        }

        const jobToSubmit = {};
        jobToSubmit.config = jobConfig;

        // TODO: get these from process.env, or from the user if provided
        jobToSubmit.dirs = {
          // A job may not have this if made from fork of another job (based on query)
          in: s3Bucket,
          out: s3Bucket
        };

        jobToSubmit.userID = userID;

        jobToSubmit.assembly = 'hg38';

        if (!jobToSubmit.name) {
          let name = jobConfig.filesSelected.data.split(' ').join('_');

          name = name.substr(0, name.lastIndexOf('.'));

          const date = new Date();

          name = name + '_' + date.getTime();

          jobToSubmit.name = name;
        }

        const jobDoc = new Jobs(jobToSubmit);

        // Otherwise "cannot read property of undefined"
        const fn = promisify(jobDoc.submitAnnotation.bind(jobDoc));

        try {
          console.info('Trying');
          const job = await fn();
          console.info('response', job);
          return job;
        } catch (err) {
          console.error(err);

          throw new DataFetchError();
        }
      }
    }
  };

  return jobResolver;
};

// module.exports = jobResolver;
