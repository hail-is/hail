import { PureComponent } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';
import Link from 'next/link';
import { PR, Issue } from './scorecard/scorecard';

const config = getConfig().publicRuntimeConfig.SCORECARD;
const DOMAIN: string = config.DOMAIN;
const USERS: string[] = config.USERS;

if (!(USERS && DOMAIN)) {
  throw new Error('Scorecard USERS and DOMAIN env variables required found');
}

const jsonURL = `${DOMAIN}/json`;

import '../styles/pages/scorecard.scss';
// TODO: This kind of thing is maybe better represented using GraphQL
// buys use schema introspection and validation
// Typescript gives us only compile-time guarantees on the client

declare type scorecardJson = {
  data: {
    user_data: {
      [name: string]: {
        CHANGES_REQUESTED: PR[];
        ISSUES: Issue[];
        NEEDS_REVIEW: PR[];
      };
    };
    unassigned: PR[];
    urgent_issues: [
      {
        AGE: string;
        ISSUE: Issue;
        USER: string;
      }
    ];
    updated: string;
  };
};

declare type user = {
  user: string;
};

interface Props {
  pageProps: scorecardJson & user;
}

// TODO: think about triggering this in _app.js
// or simply set an expiration, and then upon mount check if expiration
// is too stale
// An example of how we can cache events that can be slightly stale
// at some memory cost
// With the benefit that if refresh is smaller than the
// time between page clicks, non-stale state will be served in << 16ms on click
let cache: object;

let timeout: NodeJS.Timeout;
const startPolling = (ms: number = 1 * 60 * 1000) => {
  if (timeout) {
    clearTimeout(timeout);
  }

  timeout = setInterval(() => {
    fetch(jsonURL)
      .then(d => d.json())
      .then(data => {
        cache = data;
      });
  }, ms);
};

class Scorecard extends PureComponent<Props, scorecardJson & user> {
  // Data that is fetched during the server rendering phase does not need
  // to be re-fetched during the client rendering phase
  // The data is automatically available under this.props.pageProps
  static refreshUser() {
    const seed = Math.floor(Math.random() * USERS.length);
    return USERS[seed];
  }

  static async getInitialProps() {
    const user = Scorecard.refreshUser();

    if (typeof window === 'undefined' || !cache) {
      const ssr = await fetch(jsonURL).then(d => d.json());

      // TODO: could use page loading indicator here instead of synchronously waiting
      if (typeof window !== 'undefined') {
        cache = ssr;
        startPolling();
      }

      return { pageProps: { data: ssr, user } };
    }

    return { pageProps: { user } };
  }

  constructor(props: any) {
    super(props);

    // Initialize state to props becaues we may mutate the state (say polling)
    // but props are supposed to be read-only
    this.state = {
      data: this.props.pageProps.data || cache,
      user: this.props.pageProps.user
    };
  }

  handleRefreshUser = () => {
    this.setState({ user: Scorecard.refreshUser() });
  };

  render() {
    if (!this.state.data) {
      return <div>No data</div>;
    }

    const { user_data, unassigned, urgent_issues } = this.state.data;

    if (unassigned && unassigned.length) {
      user_data['*UNASSIGNED*'] = {
        NEEDS_REVIEW: unassigned,
        CHANGES_REQUESTED: [],
        ISSUES: []
      };
    }

    return (
      <span id="scorecard">
        <div className="issues-section">
          <h5>Nominal</h5>
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Review</th>
                <th>Change</th>
                <th>Issues</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(user_data).map((name, idx) => (
                <tr key={idx}>
                  <td>
                    <Link href={`/scorecard/user?name=${name}`}>
                      <a>{name}</a>
                    </Link>
                  </td>
                  <td>
                    {user_data[name].NEEDS_REVIEW.map((pr, i) => (
                      <a target="_blank" key={i} href={pr.html_url}>
                        {pr.id}
                      </a>
                    ))}
                  </td>
                  <td>
                    {user_data[name].CHANGES_REQUESTED.map((pr, i) => (
                      <a target="_blank" key={i} href={pr.html_url}>
                        {pr.id}
                      </a>
                    ))}
                  </td>
                  <td>
                    {user_data[name].ISSUES.map((pr, i) => (
                      <a target="_blank" key={i} href={pr.html_url}>
                        {pr.id}
                      </a>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {urgent_issues && (
          <div className="issues-section">
            <h5>Urgent !</h5>
            {
              <table>
                <thead>
                  <tr>
                    <th>Who</th>
                    <th>When</th>

                    <th>What</th>
                  </tr>
                </thead>
                <tbody>
                  {urgent_issues.map((issue, idx) => (
                    <tr key={idx}>
                      <td>
                        <Link href={`/scorecard/user?name=${issue.USER}`}>
                          <a>{issue.USER}</a>
                        </Link>
                      </td>
                      <td>{issue.AGE}</td>
                      <td>
                        <a target="_blank" href={issue.ISSUE.html_url}>
                          {issue.ISSUE.title}
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          </div>
        )}
        <div className="issues-section">
          <h5>Rolodex</h5>
          <div id="random-user">
            <img
              src={`https://github.com/${this.state.user}.png?size=50`}
              width="50"
              onClick={this.handleRefreshUser}
            />
            <span>{this.state.user}</span>
          </div>
        </div>
      </span>
    );
  }
}

export default Scorecard;
