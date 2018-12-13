import React from 'react';
import gql from 'graphql-tag';
import { Mutation } from 'react-apollo';

const test = {
  flags: {
    hwe: false,
    pca: true
  },
  options: {
    regression: 'assoc',
    hwe: '',
    geno: '',
    maf: 'tests',
    email: 'pakotlar@gmail.com'
  },
  filesSelected: {
    fam: 'someFam',
    data: 'somedata'
  },
  samplesSelected: []
};

const SUBMIT_JOB = gql`
  mutation SubmitJob($jobConfig: JobConfigInput!) {
    submitJob(jobConfig: $jobConfig) {
      config {
        options {
          maf
        }
      }
      results
    }
  }
`;

const SubmitJob = props => (
  <Mutation
    mutation={SUBMIT_JOB}
    variables={{ jobConfig: test }} //props.jobConfig
    key={1}
  >
    {(submitJob, { loading, error }) => (
      <React.Fragment>
        <button type="button" onClick={() => submitJob()}>
          Submit
        </button>
        {loading && <p>Loading...</p>}
        {error && <p>Error :( Please try again</p>}
      </React.Fragment>
    )}
  </Mutation>
);

export default SubmitJob;
