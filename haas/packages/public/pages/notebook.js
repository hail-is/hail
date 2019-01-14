import { PureComponent } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';
import Overview from '../components/Scorecard/Overview';
import User from '../components/Scorecard/User';

const { publicRuntimeConfig } = getConfig();

const { URL } = publicRuntimeConfig.NOTEBOOK;

// TODO: Finish
import 'components/Notebook/notebook.scss';

class Notebook extends PureComponent {
  // Anything you want run on the server
  static async getInitialProps(props) {
    let notebooks;
    try {
      notebooks = await fetch(URL).then(res => res.json());
    } catch (err) {
      console.error(err);
    }

    const r = {};
    if (notebooks) {
      r.notebooks = notebooks;
    }

    return r;
  }

  render() {
    if (!this.props.notebooks) {
      return (
        <span id="notebook">
          <button className="mdc-button">
            <span className="mdc-button__label">Create Notebook</span>
          </button>
        </span>
      );
    }
  }
}

export default Notebook;
