import { PureComponent } from 'react';
import { authenticationCallback } from '../libs/auth';
import Router from 'next/router';

class Redirect extends PureComponent {
  constructor(props: any) {
    if (typeof window !== 'undefined') {
      authenticationCallback(err => {
        if (err) {
          console.error(err);
          return;
        }

        Router.replace('/');
      });
    }

    super(props);
  }

  render() {
    return <div style={{ margin: 'auto' }}>Unauthorized</div>;
  }
}

export default Redirect;
