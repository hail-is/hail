// An example signup component
// This is not fully working, but provides a nice UI (needs submission)
// TODO: remove dependence on Material UI

import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import Head from 'next/head';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Button from '@material-ui/core/Button';

// why isn't this next/Link??
import Link from 'next/link';

const styles = theme => ({
  root: {
    display: 'flex',
    flexWrap: 'wrap',
    flexDirection: 'column'
  },
  formControl: {
    margin: theme.spacing.unit,
    minWidth: 150,
    maxWidth: 320
  },
  selectEmpty: {
    marginTop: theme.spacing.unit * 2
  },
  card: {
    minWidth: 150,
    maxWidth: 320
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)'
  },
  title: {
    fontSize: 14
  },
  pos: {
    marginBottom: 12
  }
});

// Need to pass props to inner <a>, from Button, because otherwise
// the className doesn't get passed and the anchor tag is not styled
// const MyLink = props => (

// );

class Signup extends React.Component {
  state = {
    isLoggedIn: false
  };

  componentDidMount() {
    // this.setState({
    //   labelWidth: ReactDOM.findDOMNode(this.InputLabelRef).offsetWidth
    // });
  }

  render() {
    const { classes } = this.props;

    return (
      <React.Fragment>
        <Head>
          <title>Sign Up</title>
        </Head>
        <Card className={classes.card}>
          {/* <form className={classes.root} autoComplete="off"> */}
          {!this.state.loggedIn ? (
            <React.Fragment>
              <CardContent style={{ textAlign: 'left' }}>
                <h3>Sign Up</h3>

                <div style={{ flexDirection: 'column', alignContent: 'left' }}>
                  {/* <TextField
                    id="login-name"
                    label="Name "
                    className={classes.textField}
                    margin="normal"
                    fullWidth
                  /> */}
                  <TextField
                    id="login-email"
                    label="Email address"
                    className={classes.textField}
                    type="email"
                    margin="normal"
                    required
                    fullWidth
                  />

                  <TextField
                    id="login-pass"
                    label="Password"
                    className={classes.textField}
                    type="password"
                    margin="normal"
                    required
                    fullWidth
                  />
                </div>
              </CardContent>

              <CardActions>
                {/* without wrapping elem or fragment Link will get className and complain
              this also prevents cardactions from adding padding to children*/}
                {/* <React.Fragment> */}
                <Button>Submit</Button>
                <span>|</span>
                {/* <Link href="/login" passHref> */}
                <Button
                  className={'test'}
                  component={props => (
                    <Link href="/login" passHref>
                      <a {...props}>{props.children}</a>
                    </Link>
                  )}
                >
                  Log In
                </Button>
                {/* </Link> */}
                {/* </React.Fragment> */}
              </CardActions>
            </React.Fragment>
          ) : (
            <React.Fragment>
              <div>You're logged in</div>
            </React.Fragment>
          )}
          {/* </form> */}
        </Card>
      </React.Fragment>
    );
  }
}

Signup.propTypes = {
  classes: PropTypes.object.isRequired
};

export default withStyles(styles)(Signup);
