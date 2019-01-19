import { PureComponent } from 'react';
import fetch from 'isomorphic-unfetch';

class Scorecard extends PureComponent {
  static async getInitialProps() {
    const data = await fetch('https://scorecard.hail.is/json').then(d =>
      d.json()
    );

    return { pageProps: { data } };
  }

  async componentDidMount() {
    // const data = await fetch('https://scorecard.hail.is/json').then(d =>
    //   d.json()
    // );
    //console.info('fetched');
  }

  render() {
    return <div>Scorecard</div>;
  }
}

export default Scorecard;
