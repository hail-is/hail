import { useState } from 'react';
import { ToggleSwitch } from '../../shared/ToggleSwitch';

const REACT_UI_COOKIE = 'hail_react_ui';
// Matches the max-age used everywhere this cookie is set (react_toggle.js, JobPage, etc).
const REACT_UI_COOKIE_MAX_AGE_SECONDS = 2147483647;

function humanizeSeconds(totalSeconds: number): string {
  const years = totalSeconds / (365.25 * 24 * 60 * 60);
  return `~${years.toFixed(0)} years`;
}

function csrfToken(): string {
  return document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') ?? '';
}

function disableReactUi() {
  document.cookie = `${REACT_UI_COOKIE}=; max-age=0; path=/; SameSite=Lax`;
  location.reload();
}

interface UserPageProps {
  authBaseUrl: string;
  username: string;
  gsaDisplayName: string;
  trialBpName: string;
}

export function UserPage({ authBaseUrl, username, gsaDisplayName, trialBpName }: UserPageProps) {
  // This page only renders when the cookie is already '1', so the toggle starts checked.
  const [reactUiEnabled, setReactUiEnabled] = useState(true);

  return (
    <div id="profile" className="vcentered space-y-2">
      <h1 className="text-4xl mb-4">{username}</h1>

      <form action={`${authBaseUrl}/logout`} method="POST">
        <input type="hidden" name="_csrf" value={csrfToken()} />
        <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
          Log out
        </button>
      </form>

      <p>
        <b>Notice:</b> By continuing to use the Hail system, you agree that you have reviewed and
        will be bound by the Hail <a href="https://batch.hail.is/tos" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Terms of Service</a> and
        have read the <a href="https://batch.hail.is/privacy" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Privacy Policy</a>. If not you
        must log out and stop using the Hail system immediately.
      </p>

      <p><b>Google Service Account: </b>{gsaDisplayName}</p>
      <p><b>Trial Billing Project: </b>{trialBpName}</p>

      <form action={`${authBaseUrl}/copy-paste-token`} method="post">
        <input type="hidden" name="_csrf" value={csrfToken()} />
        <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
          Get a copy-paste login token
        </button>
      </form>

      <div className="pt-6">
        <h2 className="text-xl font-light mb-2">
          User Settings <span className="text-xs align-middle text-amber-600 font-normal">[In Development]</span>
        </h2>
        <ToggleSwitch
          checked={reactUiEnabled}
          onChange={(next) => {
            if (next) return; // already enabled, since this page only renders when the cookie is set
            setReactUiEnabled(false);
            disableReactUi();
          }}
          label="New layout"
        />

        <table className="mt-3 text-sm border-collapse">
          <thead>
            <tr className="text-left text-zinc-500">
              <th className="pr-4 font-normal">Name</th>
              <th className="pr-4 font-normal">Scope</th>
              <th className="pr-4 font-normal">Value</th>
              <th className="pr-4 font-normal">Max age</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t">
              <td className="pr-4 py-1 font-mono">{REACT_UI_COOKIE}</td>
              <td className="pr-4 py-1 font-mono">path=/</td>
              <td className="pr-4 py-1 font-mono">1</td>
              <td className="pr-4 py-1">{humanizeSeconds(REACT_UI_COOKIE_MAX_AGE_SECONDS)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        The Hail system records your email address and IP address. Your email address
        is recorded so that we can authenticate you. Your IP address is tracked as part of our
        surveillance of all traffic to and from the Hail system. This broad surveillance enables the
        protection of the Hail system from malicious actors.
      </p>
    </div>
  );
}
