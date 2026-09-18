import { useState, ReactNode } from 'react';
import { CopyPasteToken } from './CopyPasteToken';

const REACT_UI_COOKIE = 'hail_react_ui';

function csrfToken(): string {
  return document.head.querySelector('meta[name="csrf"]')?.getAttribute('value') ?? '';
}

function disableReactUi() {
  document.cookie = `${REACT_UI_COOKIE}=; max-age=0; path=/; SameSite=Lax`;
  location.reload();
}

function Section({ title, children }: { title: string; children: ReactNode }): JSX.Element {
  return (
    <section className="pt-6">
      <h2 className="text-lg font-semibold text-zinc-700 border-b pb-1 mb-3">{title}</h2>
      {children}
    </section>
  );
}

function CopyableValue({ value }: { value: string }): JSX.Element {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    void navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => { setCopied(false); }, 1500);
    });
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="font-mono">{value}</span>
      <button onClick={handleCopy} title="Copy to clipboard" className="text-zinc-400 hover:text-zinc-700">
        {copied ? '✓' : '⧉'}
      </button>
    </span>
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
    <div id="profile" className="space-y-2">
      <Section title="User Account">
        <div className="space-y-2">
          <p><b>Logged in as: </b><CopyableValue value={username} /></p>
          <p><b>Google Service Account: </b><CopyableValue value={gsaDisplayName} /></p>
          <p><b>Trial Billing Project: </b><CopyableValue value={trialBpName} /></p>
          <form action={`${authBaseUrl}/logout`} method="POST" className="pt-2">
            <input type="hidden" name="_csrf" value={csrfToken()} />
            <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Log out
            </button>
          </form>
        </div>
      </Section>

      <Section title="Session Management">
        <CopyPasteToken authBaseUrl={authBaseUrl} />
      </Section>

      <Section title="User Settings">
        <p className="text-sm">
          <b>UI style:</b> new layout <span className="text-xs align-middle text-amber-600">[In Development]</span> ·
          <button onClick={disableReactUi} className="text-sky-600 hover:underline">Back to classic layout</button>
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
