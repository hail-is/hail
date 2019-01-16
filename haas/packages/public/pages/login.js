import { PureComponent } from 'react';
import LoginLink from 'components/Login/Link';
import Auth from 'lib/Auth';

class LoginPage extends PureComponent {
  state = {
    workshopPass: null
  };

  handleSubmit = e => {
    console.info('Handling submission');
    Auth.login(this.state.workshopPass);
    e.preventDefault();
  };

  handleChange = e => {
    this.setState({ workshopPass: e.target.value });
  };

  render() {
    return (
      <div className="card">
        <div className="card-header">
          <h4>Login</h4>
        </div>
        <div
          className="card-body"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 30
          }}
        >
          <LoginLink /> or
          <form onSubmit={this.handleSubmit}>
            <input
              type="password"
              className="outlined-input"
              onChange={this.handleChange}
              placeholder="Workshop Password"
            />
          </form>
        </div>
      </div>
    );
  }
}

export default LoginPage;
