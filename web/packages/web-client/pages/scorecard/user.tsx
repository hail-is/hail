import { PureComponent, Fragment } from 'react';
import { PR, Issue } from '../../components/Scorecard/scorecard';
import fetch from 'isomorphic-unfetch';
import getConfig from 'next/config';

const config = getConfig().publicRuntimeConfig.SCORECARD;

// We separate these because in some environments (i.e kubernetes)
// there may be an internal DNS we can take advantage of
const WEB_URL: string = config.WEB_URL;

import '../../styles/pages/scorecard/user.scss';

declare type response = {
  data: {
    CHANGES_REQUESTED: PR[];
    NEEDS_REVIEW: PR[];
    FAILING: any[];
    ISSUES: Issue[];
  };
  updated: string;
};

declare type props = {
  query: {
    name: string;
  };
};

declare type pageProps = {
  pageProps: response & {
    name: string;
  };
};

declare type section = {
  data: PR[] | Issue[];
  sectionClass: 'pr' | 'issue';
  type: string;
};

declare type prProps = {
  data: PR[];
};

declare type issueProps = {
  data: Issue[];
};

const PrSection = ({ data }: prProps) => (
  <Fragment>
    {data.map((pr: PR, idx: number) => (
      <span className="data-section" key={idx}>
        <a href={pr.html_url} target="_blank">
          {pr.id}
        </a>
        <a href={`https://github.com/${pr.user}`} target="_blank">
          {pr.user}
        </a>
      </span>
    ))}
  </Fragment>
);

const IssueSection = ({ data }: issueProps) => (
  <Fragment>
    {data.map((issue: Issue, idx: number) => (
      <span className="data-section" key={idx}>
        <a href={issue.html_url} target="_blank">
          <span>{issue.id}</span>
          <span>{issue.title}</span>
        </a>
      </span>
    ))}
  </Fragment>
);

const Section = ({ data, sectionClass, type }: section) => (
  <div className={`section ${data.length === 0 ? 'deemph' : ''}`}>
    <h5>{type}</h5>
    <div className="data">
      {data.length > 0 ? (
        sectionClass === 'pr' ? (
          <PrSection data={data as PR[]} />
        ) : (
          <IssueSection data={data as Issue[]} />
        )
      ) : (
        <div>None</div>
      )}
    </div>
  </div>
);

// TODO: add caching
class User extends PureComponent<pageProps> {
  static async getInitialProps({ query }: props) {
    const res: response = await fetch(
      `${WEB_URL}/json/users/${query.name}`
    ).then(d => d.json());

    return {
      pageProps: {
        data: res.data,
        updated: res.updated,
        name: query.name
      }
    };
  }

  constructor(props: pageProps) {
    super(props);
  }

  render() {
    const {
      data: { CHANGES_REQUESTED, NEEDS_REVIEW, FAILING, ISSUES },
      updated,
      name
    } = this.props.pageProps;

    return (
      <div id="scorecard-user">
        <a
          id="head"
          className="section"
          href={`https://github.com/${name}`}
          target="_blank"
        >
          <img src={`https://github.com/${name}.png?size=50`} width="50" />
          <h3>{name}</h3>
        </a>
        <div>
          <Section
            data={CHANGES_REQUESTED}
            sectionClass="pr"
            type="Changes Requested"
          />
          <Section data={NEEDS_REVIEW} sectionClass="pr" type="Needs Review" />
          <Section data={FAILING} sectionClass="pr" type="Failing Tests" />
          <Section data={ISSUES} sectionClass="issue" type="Issues" />
          <span className="deemph">Updated: {updated}</span>
        </div>
      </div>
    );
  }
}

export default User;
