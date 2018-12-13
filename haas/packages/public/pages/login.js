// An example login component
// This is not fully working, but provides a nice UI (needs submission)
// TODO: remove dependence on Material UI
import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Input from '@material-ui/core/Input';
import OutlinedInput from '@material-ui/core/OutlinedInput';
import FilledInput from '@material-ui/core/FilledInput';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import TextField from '@material-ui/core/TextField';
import Head from 'next/head';
import Paper from '@material-ui/core/Paper';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';

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

class Login extends React.Component {
  state = {
    isLoggedIn: false
  };

  // constructor(props) {
  //   super(props);
  // }

  componentDidMount() {
    // if (process.browser) {
    //   const auth = new Auth();
    // if (this.props.auth.isAuthenticated()) {
    //   // console.info('TRUE');
    //   // alert('expred');
    //   this.props.auth.login();
    // }
  }

  render() {
    const { classes } = this.props;

    return (
      <React.Fragment>
        <Head>
          <title>Log In</title>
        </Head>
        <Card className={classes.card}>
          {/* <form className={classes.root} autoComplete="off"> */}
          {!this.state.loggedIn ? (
            <React.Fragment>
              <CardContent style={{ textAlign: 'left' }}>
                <h3>Log In</h3>

                <div style={{ flexDirection: 'column', alignContent: 'left' }}>
                  <TextField
                    id="login-email"
                    label="Email address"
                    defaultValue={this.state.email || ''}
                    // className={classes.textField}
                    type="email"
                    margin="normal"
                    fullWidth
                  />

                  <TextField
                    id="login-pass"
                    label="Password"
                    defaultValue=""
                    // className={classes.textField}
                    type="password"
                    margin="normal"
                    fullWidth
                  />
                </div>
              </CardContent>
              <CardActions>
                {/* without wrapping elem or fragment Link will get className and complain
              this also prevents cardactions from adding padding to children*/}

                <Button>Submit</Button>
                <span>|</span>
                {/* <React.Fragment> */}
                {/* <Link href="/signup" passHref> */}
                <Button
                  component={props => (
                    <Link href="/signup" passHref>
                      <a {...props}>{props.children}</a>
                    </Link>
                  )}
                >
                  Sign Up
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

Login.propTypes = {
  classes: PropTypes.object.isRequired
};

export default withStyles(styles)(Login);
