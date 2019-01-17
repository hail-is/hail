import { PureComponent } from 'react';
import auth from '../libs/auth';

class Login extends PureComponent {
  render() {
    return (
      <span className="card">
        <span className="card-header">
          <h3>Login</h3>
        </span>
        <button className="outlined-button">Login</button>
        or
        <form onSubmit={this.handleSubmit}>
          <input type="password" />
        </form>
      </span>
    );
  }
}

export default Login;
