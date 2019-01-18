import { PureComponent } from 'react';
import auth from '../libs/auth';
import Router from 'next/router';
// import cookie from '../libs/cookies';

declare type state = {
  password?: string;
  loggedIn?: boolean;
  failed?: boolean;
};

class Login extends PureComponent {
  state: state = {};

  constructor(props: any) {
    super(props);
  }

  onLoginButtonClick = async () => {
    try {
      await auth.login(this.state.password);

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
    if (this.state.failed) {
      return <div>Unauthorized!</div>;
    }

    return (
      <span id="login">
        <h3>Login</h3>
        <span className="card">
          <button className="outlined-button" onClick={this.onLoginButtonClick}>
            Login
          </button>
        </span>
        <span className="card">
          <form onSubmit={this.onSubmit}>
            <input type="password" onChange={this.onChange} />
          </form>
        </span>
      </span>
    );
  }
}

export default Login;
