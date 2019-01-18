import { PureComponent } from 'react';
import auth from '../libs/auth';

class Callback extends PureComponent {
  state = {
    loginFailed: false
  };

  componentDidMount() {
    auth.handleCallback();
  }

  render() {
    return this.state.loginFailed ? (
      <div>Unauthorized</div>
    ) : (
      <div>Loading</div>
    );
  }
}

export default Callback;
