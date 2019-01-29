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

  handleClick = async () => {
    const formData = new FormData();
    formData.append('image', 'hail');

    const data = await fetch(`${DOMAIN}/new`, {
      headers: {
        Authorization: `Bearer ${auth.accessToken}`
      },
      method: 'POST',
      body: formData
    }).then(d => d.json());

    console.info('data', data);
  };

  render() {
    return (
      <div id="notebook" className="centered">
        <button onClick={this.handleClick}>Create</button>
      </div>
    );
  }
}

export default Notebook;
