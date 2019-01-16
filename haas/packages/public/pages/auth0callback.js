// TODO: Replace Loading... with circular loader component that isn't
// based on Material UI
import { PureComponent } from 'react';
import Router from 'next/router';
import Auth from 'lib/Auth';

class Callback extends PureComponent {
  state = {
    requestComplete: false
  };

  componentDidMount() {
    if (Auth.isAuthenticated()) {
      Router.replace('/');
      return;
    }

    Auth.handleAuthenticationAsync(err => {
      if (err) {
        this.setState({
          requestComplete: true
        });

        return;
      }

      Router.replace('/');
    });
  }

  render() {
    return !this.state.requestComplete ? (
      <div>Loading</div>
    ) : (
      <div>Unauthorized</div>
    );
  }
}

export default Callback;
