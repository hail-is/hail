// TODO: actually use user's github authenticated session
// TODO: json-pretty dep only for illustrative purposes, remove

import { Component } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';
import Overview from '../components/Scorecard/Overview';

const { publicRuntimeConfig } = getConfig();

const url = publicRuntimeConfig.SCORECARD.URL;

// TODO: Finish
class Scorecard extends Component {
  state = {
    viewedUser: null
  };

  // Anything you want run on the server
  static async getInitialProps() {
    try {
      const userData = await fetch(url).then(res => res.json());

      return { userData };
    } catch (err) {
      return {
        userData: null
      };
    }
  }

  render() {
    if (!this.state.viewedUser) {
      return <Overview data={this.props.userData} />;
    }

    return <div>Hi!</div>;
  }
}

export default Scorecard;
