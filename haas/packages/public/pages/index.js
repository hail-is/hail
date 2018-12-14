import { Fragment } from 'react';
import Head from 'next/head';
import 'animate.css';

const Index = () => (
  <Fragment>
    <Head>
      <title>Hail</title>
    </Head>

    <h2 className={'animated fadeInUp'}>
      <a className={'primary'} href="https://hail.is" target="_blank">
        Hail
      </a>
    </h2>

    {/* <div
      style={{ marginTop: 22 }}
      className={'animated fadeIn delay-1s faster'}
    >
      Speed all the things
    </div> */}
  </Fragment>
);

export default Index;
