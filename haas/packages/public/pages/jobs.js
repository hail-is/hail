// Example component, with 2 steps: select some existing submission
// say could be from batch
// Then fiew information about that job using db state
import React from 'react';
import PropTypes from 'prop-types';
import Router, { withRouter } from 'next/router';

import ChooseJobs from '../components/Jobs/ChooseJob';
import JobInfoCard from '../components/Jobs/JobInfoCard';

const handleSelect = job => {
  Router.push({
    pathname: Router.pathname,
    query: { jobID: job.id }
  });
};

//Document this
const handleClear = () => {
  Router.push({
    pathname: Router.pathname,
    query: {}
  });
};
const Jobs = ({ router: { query } }) =>
  !(query && query.jobID) ? (
    <ChooseJobs onSelected={handleSelect} />
  ) : (
    <JobInfoCard id={query.jobID} onClear={handleClear} />
  );

Jobs.propTypes = {
  router: PropTypes.shape({
    query: PropTypes.shape({
      jobID: PropTypes.string
    }).isRequired
  }).isRequired
};

export default withRouter(Jobs);
