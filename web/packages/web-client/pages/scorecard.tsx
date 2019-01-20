import { PureComponent } from 'react';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const DOMAIN = getConfig().publicRuntimeConfig.SCORECARD.DOMAIN;
const jsonURL = `${DOMAIN}/json`;

declare type scorecardJson = {
  data: object;
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

    return <div>Have data</div>;
    // <span id="scorecard">
    //   <table>
    //     <thead>
    //       <tr>User</tr>
    //       <tr>Review</tr>
    //       <tr>Change</tr>
    //       <tr>Issues</tr>
    //     </thead>
    //     <tbody>
    //       <trow>
    //       <th>User</th>
    //     </tbody>
    //   </table>
    // </span>
  }
}

export default Scorecard;
