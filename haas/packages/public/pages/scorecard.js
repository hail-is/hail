// TODO: actually use user's github authenticated session
// TODO: json-pretty dep only for illustrative purposes, remove
// likely by proxy
import { Component, Fragment } from 'react';
// import axios from 'axios';
import fetch from 'isomorphic-unfetch';
import Link from 'next/link';
import getConfig from 'next/config';
import '../styles/scorecard/scorecard.scss';

// import { ApolloClient } from 'apollo-client';
// import { ApolloLink } from 'apollo-link';
// import { HttpLink } from 'apollo-link-http';
// import { setContext } from 'apollo-link-context';
// import { InMemoryCache } from 'apollo-cache-inmemory';
// import gql from 'graphql-tag';
// import { Query } from 'react-apollo';

const urgentType = 'prio:high';

// const t = 'prio:high'; declare type urgentType = t; #t not found
//  TODO: use types from Github GraphQL api
// declare type pullRequest = {
//   repo: string; //repo name (e.g hail)
//   id: string; //number in the repository
//   title: string;
//   assignees: [string];
//   htmlUrl: string;
//   urgent?: boolean;
//   createdAt: string;
//   state: string;
//   status: string;
// };

// declare type userData = {
//   pr: pullRequest;
//   issues: any;
// };

import ErrorMessage from '../components/ErrorMessage';

// TODO: modify initApollo to allow creation of multiple singleotn endpoints
// and use that along with withApolloClient
// serverRuntimeConfig is only seen during server renderes
const { serverRuntimeConfig, publicRuntimeConfig } = getConfig();

const url = publicRuntimeConfig.SCORECARD.URL;
console.info('url', url);
// const httpLink = new HttpLink({
//   uri: 'https://api.github.com/graphql' // Server URL (must be absolute)
// });
// console.info(publicRuntimeConfig, serverRuntimeConfig);
// const authLink = setContext((request, previousContext) => {
//   // TODO: handle missing auth token
//   return {
//     headers: {
//       ...previousContext.headers,
//       authorization: `Bearer ${publicRuntimeConfig.GITHUB.ACCESS_TOKEN_UNSAFE}`
//     }
//   };
// });

// let count = 0;
// const cache = new InMemoryCache();

// // TODO: reuse, properly taking into account browser vs server side rendering mode
// const customClient = new ApolloClient({
//   ssrMode: true,
//   connectToDevTools: process.browser,
//   link: ApolloLink.from([authLink, httpLink]),
//   cache
// });

// const GITHUB_QUERY = gql`
//   query {
//     repository(owner: "hail-is", name: "hail") {
//       issues(
//         last: 100
//         states: [OPEN]
//         orderBy: { field: CREATED_AT, direction: ASC }
//       ) {
//         edges {
//           node {
//             number
//             author {
//               login
//             }
//           }
//         }
//       }
//       pullRequests(last: 100, states: [OPEN]) {
//         edges {
//           node {
//             number
//             state
//             reviews(last: 10) {
//               edges {
//                 node {
//                   state
//                   author {
//                     login
//                   }
//                   commit {
//                     id
//                     status {
//                       state
//                     }
//                   }
//                 }
//               }
//             }

//             labels(last: 10) {
//               edges {
//                 node {
//                   name
//                 }
//               }
//             }
//             author {
//               login
//             }
//             assignees(last: 1) {
//               edges {
//                 node {
//                   login
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// `;

const PullRequestLink = props =>
  props.data.map((pr, idx) => [
    idx > 0 && ', ',
    <Link href={`${pr.html_url}`} key={idx}>
      <a target="_b"> {pr.id}</a>
    </Link>
  ]);

// TODO: Finish
class Scorecard extends Component {
  // Anything you want run on the server
  static async getInitialProps(ctx) {
    console.info('url is', url);
    try {
      const userData = await fetch(url).then(res => res.json());

      return { userData };
    } catch (err) {
      console.info('err', err);
      return {
        userData: null
      };
    }
  }

  handleCompleted = data => {
    const users = {};
    const hiPrior = [];
    data.repository.issues.edges.forEach(v => {
      if (!users[v.node.author.login]) {
        users[v.node.author.login] = {};
      }
    });

    data.repository.pullRequests.edges.forEach(v => {
      // const pr: pullRequest;
      // const reviews = v.node.reviews.edges;

      v.node.reviews.edges.forEach(review => {
        if (!users[v.node.author.login]) {
          users[v.node.author.login] = {
            pullRequests: []
          };
        }
      });
    });
  };

  render() {
    const { classes, userData } = this.props;

    if (userData === null) {
      return <div>No data</div>;
    }

    const { user_data, unassigned, urgent_issues } = userData;

    return (
      <div id="scorecard" className="grid-container">
        {urgent_issues.length > 0 && (
          <div className="grid-item" style={{ marginBottom: 16 }}>
            <h3 id="urgent">Urgent</h3>

            <table>
              <thead>
                <tr>
                  <th align="left">
                    <h5>Asignee</h5>
                  </th>
                  <th align="left">
                    <h5>Time outstanding</h5>
                  </th>
                  <th align="left">
                    <h5>Issue</h5>
                  </th>
                </tr>
              </thead>
              <tbody>
                {urgent_issues.map((issue, idx) => (
                  <tr key={idx}>
                    <td align="left">
                      <a href={`/users/${issue.USER}`}>{issue.USER}</a>
                    </td>
                    <td align="left">{issue.AGE}</td>
                    <td align="left">
                      <a href={issue.ISSUE.html_url}>{issue.ISSUE.title}</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="grid-item">
          <h3>Nominal</h3>
          <table>
            <thead>
              <tr>
                <th align="left">
                  <h5>User</h5>
                </th>
                <th align="left">
                  <h5>Review</h5>
                </th>
                <th align="left">
                  <h5>Changes</h5>
                </th>
                <th align="left">
                  <h5>Issues</h5>
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(user_data).map((userName, idx) => (
                <tr key={idx}>
                  <td align="left">
                    <Link href={`/users/${userName}`}>
                      <a target="_b"> {userName}</a>
                    </Link>
                  </td>
                  <td align="left" className="link">
                    <PullRequestLink data={user_data[userName].NEEDS_REVIEW} />
                  </td>

                  <td align="left" className="link">
                    <PullRequestLink
                      data={user_data[userName].CHANGES_REQUESTED}
                    />
                  </td>
                  <td align="left" className="link">
                    <PullRequestLink data={user_data[userName].ISSUES} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginLeft: 10, marginBottom: 16 }} className="link">
          Unassigned: <PullRequestLink data={unassigned} />
        </div>
      </div>
    );
  }
}

export default Scorecard;
