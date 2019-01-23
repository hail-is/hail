import React, { memo } from 'react';

const index = memo(() => (
  <span className="centered">
    <h3 className={'animated fadeInUp faster'}>
      <a href="https://hail.is/" target="_blank">
        Hail
      </a>
    </h3>
    {/* <span className={'animated fadeInUp faster'}>All the things</span> */}
  </span>
));

export default index;
