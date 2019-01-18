import { PureComponent } from 'react';
import auth from '../libs/auth';

class Login extends PureComponent {
  state = {
    password: null
  };

  onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    this.setState({
      password: e.target.value
    });
  };

  onSubmit = () => {
    auth.login(this.password);
  };

  render() {
    return (
      <span id="login">
        <h3>Login</h3>
        <span className="card">
          <button className="outlined-button">Login</button>
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
