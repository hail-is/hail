import React, { Fragment } from "react";

export default () => {
  return (
    <Fragment>
      <div />
      <md-button className="link" href="/submit">
        Start
      </md-button>
      <md-button className="link" href="/help">
        Guide
      </md-button>
      <md-button className="link" href="/public">
        Try
      </md-button>
    </Fragment>
  );
};
