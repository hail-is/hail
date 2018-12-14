// TODO: Convert to TypeScript, add prop validation
import React from 'react';
import Link from 'next/link';
import Router from 'next/router';
import './scorecard.scss';

const PullRequestTable = props =>
  props.data.length > 0 ? (
    <table>
      <tbody>
        {props.data.map((pr, idx) => (
          <tr key={idx}>
            <td>
              <a href={pr.html_url} target="_blank" rel="noopener noreferrer">
                {pr.id}
              </a>
            </td>
            <td>{pr.title}</td>
          </tr>
        ))}
      </tbody>
    </table>
  ) : (
    <p>No {props.title}</p>
  );

const User = ({ userName, data }) => {
  if (!data) {
    return <div>No data</div>;
  }

  const userData = data.data;
  const updated = data.updated;

  return (
    <div id="scorecard" className="grid-container">
      <div className="grid-item">
        <p>
          <img id="avatar" src={`https://github.com/${userName}.png?s=88`} />
        </p>
      </div>
      <div className="grid-item">
        <p>Welcome to Scorecard, {userName}!</p>
      </div>

      <h4>Needs Review</h4>
      {userData.NEEDS_REVIEW.length > 0 ? (
        <table>
          <tbody>
            {userData.NEEDS_REVIEW.map((pr, idx) => (
              <tr key={idx}>
                <td>
                  <a
                    href={pr.html_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {pr.id}
                  </a>
                </td>
                <td>
                  <a
                    href={`https://github.com/${userName}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {pr.user}
                  </a>
                </td>
                <td>{pr.title}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No reviews needed.</p>
      )}

      <h4>Changes Requested</h4>
      <PullRequestTable
        data={userData.CHANGES_REQUESTED}
        title="changes requested"
      />
      <h4>Failing tests</h4>
      <PullRequestTable data={userData.FAILING} title="failing builds" />
      <h4>Issues</h4>
      <PullRequestTable data={userData.ISSUES} title="issues" />
      <p>
        <small>last updated {updated}</small>
      </p>
    </div>
  );
};

export default User;
