import { useState } from 'react';

interface Props { code: string; maxHeight?: string }

export function CodeBlock({ code, maxHeight = '24rem' }: Props): JSX.Element {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    void navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => { setCopied(false); }, 1500);
    });
  }

  return (
    <div className="relative bg-slate-50 border rounded">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 text-xs text-zinc-400 hover:text-zinc-700 px-2 py-1 rounded bg-white border border-zinc-200"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
      <div className="overflow-auto p-2" style={{ maxHeight }}>
        <pre className="text-sm whitespace-pre-wrap break-all">{code}</pre>
      </div>
    </div>
  );
}
