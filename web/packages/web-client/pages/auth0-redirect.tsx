import { PureComponent } from 'react';
import { authenticationCallback } from '../libs/auth';
import Router from 'next/router';

class Redirect extends PureComponent {
  state = {
    unauthorized: false
  };

  constructor(props: any) {
    super(props);

    if (typeof window !== 'undefined') {
      authenticationCallback(err => {
        if (err) {
          console.error(err);
          this.state.unauthorized = true;
          return;
        }

        Router.replace('/');
      });
    }
  }

  render() {
    return (
      <div className="centered">
        {this.state.unauthorized ? 'Unauthorized' : 'Securing stuff...'}
      </div>
    );
  }
}

export default Redirect;
