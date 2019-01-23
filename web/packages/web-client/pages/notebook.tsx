import { PureComponent } from 'react';
import auth from '../libs/auth';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const DOMAIN = getConfig().publicRuntimeConfig.NOTEBOOK.DOMAIN;

class Notebook extends PureComponent {
  state = {
    unauthorized: false
  };

  static getInitialProps() {
    fetch(DOMAIN, {
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
    return (
      <div id="notebook" className="centered">
        <button>Create</button>
      </div>
    );
  }
}

export default Notebook;
