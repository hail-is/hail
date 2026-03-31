import { CodeBlock } from './CodeBlock';

type Spec = {
  image?: string;
  command?: string[];
  user_code?: string;
  resources?: Record<string, unknown>;
  env?: { name: string; value: string }[];
  input_files?: [string, string][];
  output_files?: [string, string][];
};

type Props = {
  spec: Spec | null;
  jobStr: string;
};

type SubTab = 'input' | 'main' | 'output';

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: 'input', label: 'Input' },
  { id: 'main', label: 'Main' },
  { id: 'output', label: 'Output' },
];

function FilesTable({ files }: { files: [string, string][] }): JSX.Element {
  return (
    <table className="text-sm w-full border-collapse">
      <thead>
        <tr className="text-left text-zinc-500">
          <th className="py-1 pr-4">From</th>
          <th className="py-1">To</th>
        </tr>
      </thead>
      <tbody>
        {files.map(([from, to], i) => (
          <tr key={i} className="border-t">
            <td className="py-1 pr-4 font-mono text-xs break-all">{from}</td>
            <td className="py-1 font-mono text-xs break-all">{to}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function JobSpecPanel({
  spec,
  jobStr,
  activeSubTab,
  setActiveSubTab,
}: Props & { activeSubTab: SubTab; setActiveSubTab: (t: SubTab) => void }): JSX.Element {
  return (
    <div>
      <div className="flex border-b text-base overflow-auto bg-white mb-4">
        {SUB_TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActiveSubTab(id)}
            className={`px-4 pt-3 pb-2 hover:opacity-100 border-b-2 ${
              activeSubTab === id
                ? 'border-black font-medium'
                : 'border-transparent opacity-50'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {activeSubTab === 'input' && (
        <div className="py-2">
          {spec?.input_files && spec.input_files.length > 0 ? (
            <FilesTable files={spec.input_files} />
          ) : (
            <div className="text-zinc-400 text-sm">No input files.</div>
          )}
        </div>
      )}

      {activeSubTab === 'main' && (
        <div className="space-y-4 py-2">
          {spec ? (
            <>
              {spec.image && spec.image !== '[jvm]' && (
                <div>
                  <h3 className="text-base font-medium text-zinc-600 mb-1">Image</h3>
                  <div className="font-mono text-sm break-all">{spec.image}</div>
                </div>
              )}
              {spec.user_code && (
                <div>
                  <h3 className="text-base font-medium text-zinc-600 mb-1">User Code</h3>
                  <CodeBlock code={spec.user_code} />
                </div>
              )}
              {spec.command && (
                <div>
                  <h3 className="text-base font-medium text-zinc-600 mb-1">Command</h3>
                  <CodeBlock code={spec.command.join("' '")} />
                </div>
              )}
              {spec.resources && Object.keys(spec.resources).length > 0 && (
                <div>
                  <h3 className="text-base font-medium text-zinc-600 mb-1">Resources</h3>
                  <table className="text-sm border-collapse w-full max-w-sm">
                    <tbody>
                      {Object.entries(spec.resources).map(([k, v]) => (
                        <tr key={k} className="border-t">
                          <td className="py-1 pr-4 text-zinc-500">{k}</td>
                          <td className="py-1">{String(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {spec.env && spec.env.length > 0 && (
                <div>
                  <h3 className="text-base font-medium text-zinc-600 mb-1">Environment</h3>
                  <table className="text-sm border-collapse w-full">
                    <tbody>
                      {spec.env.map(({ name, value }) => (
                        <tr key={name} className="border-t">
                          <td className="py-1 pr-4 font-mono text-xs">{name}</td>
                          <td className="py-1 font-mono text-xs break-all">{value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          ) : (
            <div className="text-zinc-400 text-sm">No spec available.</div>
          )}
          <div>
            <h3 className="text-base font-medium text-zinc-600 mb-1">Full Job JSON</h3>
            <CodeBlock code={jobStr} />
          </div>
        </div>
      )}

      {activeSubTab === 'output' && (
        <div className="py-2">
          {spec?.output_files && spec.output_files.length > 0 ? (
            <FilesTable files={spec.output_files} />
          ) : (
            <div className="text-zinc-400 text-sm">No output files.</div>
          )}
        </div>
      )}
    </div>
  );
}
