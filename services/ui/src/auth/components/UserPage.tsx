import { ReactNode } from 'react';
import { CopyPasteToken } from './CopyPasteToken';
import { CopyableValue } from './CopyableValue';

const REACT_UI_COOKIE = 'hail_react_ui';

function csrfToken(): string {
  return document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') ?? '';
}

function disableReactUi() {
  document.cookie = `${REACT_UI_COOKIE}=; max-age=0; path=/; SameSite=Lax`;
  location.reload();
}

function logout(authBaseUrl: string) {
  const form = document.createElement('form');
  form.method = 'POST';
  form.action = `${authBaseUrl}/logout`;
  const csrfInput = document.createElement('input');
  csrfInput.type = 'hidden';
  csrfInput.name = '_csrf';
  csrfInput.value = csrfToken();
  form.appendChild(csrfInput);
  document.body.appendChild(form);
  form.submit();
}

function Section({ title, children }: { title: string; children: ReactNode }): JSX.Element {
  return (
    <section className="pt-6">
      <h2 className="text-lg font-semibold text-zinc-700 border-b pb-1 mb-3">{title}</h2>
      {children}
    </section>
  );
}

interface UserPageProps {
  authBaseUrl: string;
  username: string;
  gsaDisplayName: string;
  trialBpName: string;
}

export function UserPage({ authBaseUrl, username, gsaDisplayName, trialBpName }: UserPageProps): JSX.Element {
  return (
    <div id="profile" className="vcentered space-y-2">
      <Section title="User Account">
        <div className="space-y-2">
          <p><b>Logged in as: </b><CopyableValue value={username} /></p>
          <p className="pl-4 text-sm text-zinc-500">
            – <button onClick={() => { logout(authBaseUrl); }} className="text-sky-600 hover:underline">Log out</button>
          </p>
          <p><b>Google Service Account: </b><CopyableValue value={gsaDisplayName} /></p>
          <p><b>Trial Billing Project: </b><CopyableValue value={trialBpName} /></p>
        </div>
      </Section>

      <Section title="Session Management">
        <p><b>Copy paste token: </b><CopyPasteToken authBaseUrl={authBaseUrl} /></p>
      </Section>

      <Section title="User Settings">
        <p className="text-sm">
          <b>UI style:</b> new layout <span className="text-xs align-middle text-amber-600">[In Development]</span>{' '}
          · <button onClick={disableReactUi} className="text-sky-600 hover:underline">Back to classic layout</button>
        </p>
      </Section>

      <Section title="Notices">
        <ul className="list-disc pl-5 space-y-2 text-sm text-zinc-600">
          <li>
            By continuing to use the Hail system, you agree that you have reviewed and
            will be bound by the Hail <a href="https://batch.hail.is/tos" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Terms of Service</a> and
            have read the <a href="https://batch.hail.is/privacy" target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Privacy Policy</a>. If not you
            must log out and stop using the Hail system immediately.
          </li>
          <li>
            The Hail system records your email address and IP address. Your email address
            is recorded so that we can authenticate you. Your IP address is tracked as part of our
            surveillance of all traffic to and from the Hail system. This broad surveillance enables the
            protection of the Hail system from malicious actors.
          </li>
        </ul>
      </Section>
    </div>
  );
}
