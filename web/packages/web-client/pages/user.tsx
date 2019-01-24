import { Component } from 'react';
import auth from '../libs/auth';

import '../styles/pages/user.scss';

class UserProfile extends Component {
  state: any = {
    loggedIn: true
  };

  constructor(props: any) {
    super(props);
  }

  shouldComponentUpdate(_: any, state: any) {
    return this.state.loggedIn !== state.loggedIn;
  }

  render() {
    return (
      <div id="user" className="centered">
        <span
          style={{
            flexDirection: 'row',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <img
            src={auth.user!.picture}
            width="50"
            height="50"
            style={{ marginRight: 14 }}
          />
          <h3>{auth.user!.name}</h3>
        </span>
      </div>
    );
  }
}

export default UserProfile;
