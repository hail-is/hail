import { useState } from 'react';

export function CopyableValue({ value }: { value: string }): JSX.Element {
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
