import { PureComponent } from 'react';
import auth from '../libs/auth';

class Notebook extends PureComponent {
  state = {
    unauthorized: false
  };

  static getInitialProps() {
    if (!auth.isAuthenticated()) {
    }
  }

  componentDidMount = () => {
    console.info('Notebook mounted');
  };

  render() {
    return <button>Create</button>;
  }
}

export default Notebook;
