// TODO: actually use user's github authenticated session
// TODO: json-pretty dep only for illustrative purposes, remove

import { Component } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';
import Overview from '../components/Scorecard/Overview';
import User from '../components/Scorecard/User';

const { publicRuntimeConfig } = getConfig();

const { URL, USER_URL } = publicRuntimeConfig.SCORECARD;

if (USER_URL.charAt(USER_URL.length - 1) === '/') {
  USER_URL.slice(0, -1);
}

// TODO: Finish
class Scorecard extends Component {
  // Anything you want run on the server
  static async getInitialProps(props) {
    const user = props.query && props.query.u;

    let userData;

    try {
      if (user) {
        userData = await fetch(`${USER_URL}/${user}`).then(res => res.json());
      } else {
        userData = await fetch(URL).then(res => res.json());
      }
    } catch (err) {
      console.info('err fetch in scorecard', err);
    }

    const r = {};
    if (userData) {
      r.userData = userData;
      r.user = user;
    }

    return r;
  }

  render() {
    if (!this.props.userData) {
      return <div>No data</div>;
    }

    if (!this.props.user) {
      return <Overview data={this.props.userData} />;
    }

    return <User userName={this.props.user} data={this.props.userData} />;
  }
}

export default Scorecard;
