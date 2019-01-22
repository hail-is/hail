import { PureComponent, Fragment } from 'react';
import { login } from '../libs/auth';
import Router from 'next/router';
// import cookie from '../libs/cookies';
import '../styles/pages/login.scss';

declare type state = {
  password?: string;
  loggedIn?: boolean;
  failed?: boolean;
};

interface LoginProps {
  redirected: boolean;
}

class Login extends PureComponent<LoginProps> {
  state: state = {};

  constructor(props: any) {
    super(props);
  }

  onLoginButtonClick = async () => {
    try {
      await login(this.state.password);

      Router.replace('/');
    } catch (e) {
      console.error(e);
      this.setState({ failed: true });
    }
  };

  onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    this.setState({
      password: e.target.value
    });
  };

  onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    this.onLoginButtonClick();
  };

  render() {
    return (
      <span id="login">
        {this.state.failed ? (
          <div>Unauthorized!</div>
        ) : (
          <Fragment>
            <h3>Login</h3>
            <span id="login-choices">
              <button
                className="outlined-button"
                onClick={this.onLoginButtonClick}
              >
                Login
              </button>
              <span id="separator">or</span>
              <form onSubmit={this.onSubmit}>
                <input
                  type="password"
                  onChange={this.onChange}
                  placeholder="Enter the workshop password"
                />
              </form>
            </span>
          </Fragment>
        )}
      </span>
    );
  }
}

export default Login;
