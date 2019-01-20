import { PureComponent } from 'react';
import auth from '../libs/auth';
import fetch from 'isomorphic-unfetch';

const URL = 'http://localhost:8000/notebook';

class Notebook extends PureComponent {
  state = {
    unauthorized: false
  };

  static getInitialProps() {
    fetch(URL, {
      headers: {
        Authorization: `Bearer ${auth.accessToken}`
      }
    })
      .then(d => d.json())
      .then(data => {
        console.info('yep', data);
      });
  }

  componentDidMount = () => {
    console.info('Notebook mounted');
  };

  render() {
    return <button>Create</button>;
  }
}

export default Notebook;
