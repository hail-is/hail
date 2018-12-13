// TODO: Replace Loading... with circular loader component that isn't
// based on Material UI
import { Component } from 'react';
import Router from 'next/router';
import { view } from 'react-easy-state';
import Auth from '../lib/Auth';

class Callback extends Component {
  componentDidMount() {
    Auth.handleAuthenticationAsync(err => {
      // TODO: notify in modal if error
      if (err) {
        console.error('ERROR in callback!', err);
      }

      Router.push('/');
    });
  }

  render() {
    return !Auth.isAuthenticated() ? <div>Loading</div> : <div>Hello</div>;
  }
}

export default view(Callback);
