import React from 'react';
import ReactDom from 'react-dom';

import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';

import getConfig from 'next/config';

import Grid from '@material-ui/core/Grid';
import { Query } from 'react-apollo';
import ErrorMessage from '../ErrorMessage';
import CircularProgress from '@material-ui/core/CircularProgress';

import gql from 'graphql-tag';
// import { FlatButton } from '@material-ui/core/FlatButton';
import IconButton from '@material-ui/core/IconButton';
// import { NextScript } from 'next/document';

import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';

import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import axios from 'axios';

import Snackbar from '@material-ui/core/Snackbar';
import CloudDownload from '@material-ui/icons/CloudDownload';
import ArrowBackIos from '@material-ui/icons/ArrowBackIos';

import Auth from '../../lib/Auth';

const { publicRuntimeConfig } = getConfig();
const url = publicRuntimeConfig.API_DOWNLOAD_URL;

const GET_JOB = gql`
  query job($id: ID!) {
    job(id: $id) {
      id
      createdAt
      name
      assembly
      updatedAt
      submission {
        state
        attempts
        log {
          progress
          messages
        }
      }
    }
  }
`;

const styles = {
  section: {
    margin: '20px 0 20px 0'
  },
  divider: {
    margin: 0
  }
};

class InfoCard extends React.Component {
  constructor(props) {
    super(props);
    this.downloadButtonRef = React.createRef();
    this.state = { notFound: false };
    this.timeout = null;
  }

  gogoGadgetDownload = () => {
    axios
      .get(`${url}/getSignedUrl/${this.props.id}`, {
        headers: {
          authorization: `Bearer ${Auth.getAccessToken()}`
        }
      })
      .then(res => {
        // console.info(res);
        const url = res.data;
        console.info(url);

        const node = ReactDom.findDOMNode(this.downloadButtonRef.current);

        node.setAttribute('href', url);
        node.click();
      })
      .catch(err => {
        let reason = err.response.data;

        this.setState({ notFound: reason });

        if (this.timeout) {
          clearTimeout(this.timeout);
        }

        this.timeout = setTimeout(() => {
          this.setState({ notFound: null });
        }, 1000);
      });
  };

  componentWillUnmount = () => {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
  };

  render() {
    const { classes, id, onClear } = this.props;

    return (
      <React.Fragment>
        <Query query={GET_JOB} variables={{ id }}>
          {({ loading, error, data, fetchMore }) => {
            if (loading) return <CircularProgress />;
            if (error) return <ErrorMessage error={error} />;
            if (!data.job) return <div>No job found</div>;

            const fetchedJob = data.job;
            let date = new Date(Number(fetchedJob.createdAt));
            date = date.toDateString();

            return (
              <Grid
                container
                // direction="column"
                // alignItems="center"
                // // justify="space-evenly"
                // alignContent="center"
                justify="center"
                // alignItems="center"
                // style={{ width: "100vw" }}
                spacing={24}
              >
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h5">
                        <IconButton
                          size="small"
                          onClick={onClear}
                          style={{
                            marginLeft: -12,
                            marginRight: 5,
                            marginTop: -4
                          }}
                        >
                          <ArrowBackIos fontSize="small" />
                        </IconButton>
                        {fetchedJob.name}
                      </Typography>
                      <Typography variant="subtitle2" gutterBottom>
                        Created On: {date}
                      </Typography>
                      <Typography
                        variant="subtitle2"
                        style={{ marginBottom: 20 }}
                      >
                        Status: <b>{fetchedJob.submission.state}</b>
                      </Typography>
                      <Divider variant="middle" className={classes.divider} />
                      <div className={classes.section}>
                        <Typography variant="h5">Log</Typography>
                        {fetchedJob.submission.log.messages.map((msg, idx) => {
                          return <p key={idx}>{msg}</p>;
                        })}
                      </div>
                    </CardContent>
                    {fetchedJob.submission.state === 'completed' ? (
                      <React.Fragment>
                        <Divider variant="middle" className={classes.divider} />
                        <CardActions>
                          <IconButton onClick={this.gogoGadgetDownload}>
                            <CloudDownload />
                          </IconButton>
                          <a
                            style={{ display: 'none' }}
                            ref={this.downloadButtonRef}
                            download
                          />
                        </CardActions>
                      </React.Fragment>
                    ) : null}
                  </Card>
                </Grid>
                {/* <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4">Log</Typography>
                      {fetchedJob.submission.log.messages.map(msg => {
                        return <p>{msg}</p>;
                      })}
                    </CardContent>
                  </Card>
                </Grid> */}
              </Grid>
            );
          }}
        </Query>
        <Snackbar
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'left'
          }}
          open={this.state.notFound}
          autoHideDuration={6000}
          onClose={this.handleClose}
          ContentProps={{
            'aria-describedby': 'message-id'
          }}
          message={<span id="message-id">{this.state.notFound}</span>}
        />
      </React.Fragment>
    );
  }
}

InfoCard.propTypes = {
  id: PropTypes.string.isRequired,
  classes: PropTypes.object.isRequired,
  onClear: PropTypes.func.isRequired
  // job: PropTypes.shape({
  //   createdAt: PropTypes.string.isRequired,
  //   id: PropTypes.string.isRequired
  // }).isRequired
};

export default withStyles(styles)(InfoCard);
