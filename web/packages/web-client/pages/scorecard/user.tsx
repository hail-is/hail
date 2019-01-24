import { PureComponent } from 'react';
import { PR, Issue } from './scorecard';
import fetch from 'isomorphic-unfetch';

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

declare type prInput = {
  prs: PR[];
  type: string;
};

declare type issueInput = {
  issues: Issue[];
  type: string;
};

const PrSection = ({ prs, type }: prInput) => (
  <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 14 }}>
    <h5>{type}</h5>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {prs.length ? (
        prs.map((pr: PR, idx: number) => (
          <span style={{ flexDirection: 'row' }} key={idx}>
            <a href={`pr.html_url`} target="_blank" style={{ marginRight: 14 }}>
              {pr.id}
            </a>
            <a href={`https://github.com/users/${pr.user}`}>{pr.user}</a>
          </span>
        ))
      ) : (
        <div>None</div>
      )}
    </div>
  </div>
);

const IssueSection = ({ issues, type }: issueInput) => (
  <div style={{ display: 'flex', flexDirection: 'column', marginBottom: 14 }}>
    <h5>{type}</h5>
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {issues.length ? (
        issues.map((issue: Issue, idx: number) => (
          <span style={{ flexDirection: 'row' }} key={idx}>
            <a href={`pr.html_url`} target="_blank" style={{ marginRight: 14 }}>
              {issue.id}
            </a>
            <span>{issue.title}</span>
          </span>
        ))
      ) : (
        <div>No issues</div>
      )}
    </div>
  </div>
);

// TODO: add caching
class User extends PureComponent<pageProps> {
  static async getInitialProps({ query }: props) {
    const res: response = await fetch(
      `https://scorecard.hail.is/json/users/${query.name}`
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

    console.info(props.pageProps);
  }

  render() {
    const {
      data: { CHANGES_REQUESTED, NEEDS_REVIEW, FAILING, ISSUES },
      updated,
      name
    } = this.props.pageProps;

    return (
      <div id="scorecard/user">
        <div
          style={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 14
          }}
        >
          <img
            src={`https://github.com/${name}.png?size=50`}
            width="50"
            style={{ marginRight: 14 }}
          />
          <h3>{name}</h3>
        </div>
        <div>
          <PrSection prs={CHANGES_REQUESTED} type="Changes Requested" />
          <PrSection prs={NEEDS_REVIEW} type="Needs Review" />
          <PrSection prs={FAILING} type="Failing Tests" />
          <IssueSection issues={ISSUES} type="Issues" />
          <span>Updated: {updated}</span>
        </div>
      </div>
    );
  }
}

export default User;
