import { PureComponent } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const DOMAIN = getConfig().publicRuntimeConfig.SCORECARD.DOMAIN;
const jsonURL = `${DOMAIN}/json`;

import '../styles/pages/scorecard.scss';
// TODO: This kind of thing is maybe better represented using GraphQL
// buys use schema introspection and validation
// Typescript gives us only compile-time guarantees on the client
declare type pr = {
  assignees: Array<string>;
  html_url: string;
  id: string;
  repo: string;
  state: string;
  status: string;
  title: string;
  user: string;
};

declare type issue = {
  assignees: [string];
  created_at: string;
  html_url: string;
  id: string;
  repo: string;
  title: string;
  urgent: boolean;
};

declare type scorecardJson = {
  data: {
    user_data: {
      [name: string]: {
        CHANGES_REQUESTED: [pr?];
        ISSUES: [issue?];
        NEEDS_REVIEW: [pr?];
      };
    };
    unassigned: [pr];
    urgent_issues: [
      {
        AGE: string;
        ISSUE: issue;
        USER: string;
      }
    ];
  };
};

interface Props {
  pageProps: scorecardJson;
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

class Scorecard extends PureComponent<Props, scorecardJson> {
  // Data that is fetched during the server rendering phase does not need
  // to be re-fetched during the client rendering phase
  // The data is automatically available under this.props.pageProps
  static async getInitialProps() {
    if (typeof window === 'undefined' || !cache) {
      const ssr = await fetch(jsonURL).then(d => d.json());

      // TODO: could use page loading indicator here instead of synchronously waiting
      if (typeof window !== 'undefined') {
        cache = ssr;
        startPolling();
      }

      return { pageProps: { data: ssr } };
    }

    return { pageProps: null };
  }

  constructor(props: any) {
    super(props);

    this.state = {
      data: (this.props.pageProps && this.props.pageProps.data) || cache
    };
  }

  render() {
    if (!this.state.data) {
      return <div>No data</div>;
    }

    const { user_data, unassigned, urgent_issues } = this.state.data;

    if (unassigned && unassigned.length) {
      user_data['UNASSIGNED'] = {
        NEEDS_REVIEW: unassigned,
        CHANGES_REQUESTED: [],
        ISSUES: []
      };
    }

    return (
      <span id="scorecard">
        <div className="issues-section">
          <h4>Nominal</h4>
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
                  <td>{name}</td>
                  <td>
                    {user_data[name].NEEDS_REVIEW.map((pr, i) => (
                      <a key={i} href={pr.html_url}>
                        {pr.id}
                      </a>
                    ))}
                  </td>
                  <td>
                    {user_data[name].CHANGES_REQUESTED.map((pr, i) => (
                      <a key={i} href={pr.html_url}>
                        {pr.id}
                      </a>
                    ))}
                  </td>
                  <td>
                    {user_data[name].ISSUES.map((pr, i) => (
                      <a key={i} href={pr.html_url}>
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
            <h4>Urgent</h4>
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
                        <a href={`/scorecard/users/${issue.USER}`}>
                          {issue.USER}
                        </a>
                      </td>
                      <td>{issue.AGE}</td>
                      <td>
                        <a href={issue.ISSUE.html_url}>{issue.ISSUE.title}</a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            }
          </div>
        )}
      </span>
    );
  }
}

export default Scorecard;
