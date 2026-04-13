import { useState, ReactNode } from 'react';
import { CodeBlock } from './CodeBlock';
import { JobSpec } from './jobModels';

function Unset(): JSX.Element {
  return <span className="text-zinc-300 text-xs">[unset]</span>;
}

function AsDefault({ value }: { value: React.ReactNode }): JSX.Element {
  return <span className="text-zinc-400 text-xs">[{value}]</span>;
}

function CollapsibleSection({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }): JSX.Element {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-base font-medium text-zinc-600 mb-1"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-zinc-400">
          {open
            ? <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
            : <path fillRule="evenodd" d="M8.22 5.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 10 8.22 6.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
          }
        </svg>
        {title}
      </button>
      {open && <div className="space-y-4">{children}</div>}
    </div>
  );
}

interface Props {
  spec: JobSpec | null;
  attributes?: Record<string, string>;
  instColl?: string;
}

type SubTab = 'input' | 'main' | 'output';

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: 'input', label: 'Input' },
  { id: 'main', label: 'Main' },
  { id: 'output', label: 'Output' },
];

function FilesTable({ files }: { files: { from: string; to: string }[] }): JSX.Element {
  return (
    <table className="text-sm w-full border-collapse">
      <thead>
        <tr className="text-left text-zinc-500">
          <th className="py-1 pr-4">From</th>
          <th className="py-1">To</th>
        </tr>
      </thead>
      <tbody>
        {files.map((file, i) => (
          <tr key={i} className="border-t">
            <td className="py-1 pr-4 font-mono text-xs break-all">{file.from}</td>
            <td className="py-1 font-mono text-xs break-all">{file.to}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function JobSpecPanel({
  spec,
  attributes,
  instColl,
  activeSubTab,
  setActiveSubTab,
}: Props & { activeSubTab: SubTab; setActiveSubTab: (t: SubTab) => void }): JSX.Element {
  const res = spec?.resources ?? {};
  const reqCpu = res.req_cpu as string | undefined;
  const reqMemory = res.req_memory as string | undefined;
  const reqStorage = res.req_storage as string | undefined;
  const requestedMachineType = res.machine_type as string | undefined;
  const preemptible = res.preemptible as boolean | undefined;

  return (
    <div>
      <div className="flex border-b text-base overflow-auto bg-white mb-4">
        {SUB_TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => { setActiveSubTab(id); }}
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
        <div className="space-y-4 py-2">
          <div>
            <h3 className="text-base font-medium text-zinc-600 mb-1">Input Files</h3>
            {spec?.input_files && spec.input_files.length > 0 ? (
              <FilesTable files={spec.input_files} />
            ) : (
              <div className="text-zinc-400 text-sm">None.</div>
            )}
          </div>
          {spec?.cloudfuse && spec.cloudfuse.length > 0 && (
            <div>
              <h3 className="text-base font-medium text-zinc-600 mb-1">Cloud Fuse Mounts</h3>
              <table className="text-sm w-full border-collapse">
                <thead>
                  <tr className="text-left text-zinc-500">
                    <th className="py-1 pr-4">Bucket</th>
                    <th className="py-1 pr-4">Mount path</th>
                    <th className="py-1">Read only</th>
                  </tr>
                </thead>
                <tbody>
                  {spec.cloudfuse.map((cf, i) => (
                    <tr key={i} className="border-t">
                      <td className="py-1 pr-4 font-mono text-xs break-all">{cf.bucket}</td>
                      <td className="py-1 pr-4 font-mono text-xs">{cf.mount_path}</td>
                      <td className="py-1 text-xs">{cf.read_only ? 'yes' : 'no'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeSubTab === 'main' && (
        <div className="space-y-4 py-2">
          {spec?.process ? (
            <>
              {spec.process.type === 'docker' && spec.process.image && (
                <CollapsibleSection title="Image" defaultOpen>
                  <div className="font-mono text-sm break-all">{spec.process.image}</div>
                </CollapsibleSection>
              )}
              {spec.user_code && (
                <CollapsibleSection title="User Code" defaultOpen>
                  <CodeBlock code={spec.user_code} />
                </CollapsibleSection>
              )}
              {spec.process.command && (
                <CollapsibleSection title="Command" defaultOpen>
                  <CodeBlock code={spec.process.command.join("' '")} />
                </CollapsibleSection>
              )}
              {spec.process.type === 'jvm' && spec.process.jar_spec && (
                <CollapsibleSection title="JAR" defaultOpen>
                  <div className="font-mono text-sm">{spec.process.jar_spec.type}: {spec.process.jar_spec.value}</div>
                </CollapsibleSection>
              )}
              {(reqCpu != null || reqMemory != null || reqStorage != null || requestedMachineType != null) && (
                <CollapsibleSection title="Resources" defaultOpen>
                  <table className="text-sm border-collapse w-full max-w-sm">
                    <tbody>
                      {requestedMachineType != null ? (
                        <tr className="border-t">
                          <td className="py-1 pr-4 text-zinc-500">Machine type</td>
                          <td className="py-1 font-mono">{requestedMachineType}</td>
                        </tr>
                      ) : (
                        <>
                          {reqCpu != null && (
                            <tr className="border-t">
                              <td className="py-1 pr-4 text-zinc-500">CPU</td>
                              <td className="py-1">{reqCpu}</td>
                            </tr>
                          )}
                          {reqMemory != null && (
                            <tr className="border-t">
                              <td className="py-1 pr-4 text-zinc-500">Memory</td>
                              <td className="py-1">{reqMemory}</td>
                            </tr>
                          )}
                        </>
                      )}
                      {reqStorage != null && (
                        <tr className="border-t">
                          <td className="py-1 pr-4 text-zinc-500">Storage</td>
                          <td className="py-1">{reqStorage}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </CollapsibleSection>
              )}
              <CollapsibleSection title="Settings" defaultOpen>
                <table className="text-sm border-collapse w-full max-w-sm">
                  <tbody>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Always run</td>
                      <td className="py-1">{spec.always_run != null ? (!spec.always_run ? <AsDefault value="false" /> : 'true') : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Max attempts</td>
                      <td className="py-1">{spec.n_max_attempts != null ? (spec.n_max_attempts === 20 ? <AsDefault value={20} /> : spec.n_max_attempts) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Preemptible</td>
                      <td className="py-1">{preemptible != null ? String(preemptible) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Regions</td>
                      <td className="py-1">{spec.regions ? spec.regions.join(', ') : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Timeout</td>
                      <td className="py-1">{spec.timeout != null ? `${spec.timeout}s` : <AsDefault value="no timeout" />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Always copy output</td>
                      <td className="py-1">{spec.always_copy_output != null ? String(spec.always_copy_output) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Requester pays</td>
                      <td className="py-1">{spec.requester_pays_project ?? <AsDefault value="none" />}</td>
                    </tr>
                  </tbody>
                </table>
              </CollapsibleSection>
              {spec.env && spec.env.length > 0 && (
                <CollapsibleSection title="Environment">
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
                </CollapsibleSection>
              )}
              {attributes && Object.keys(attributes).length > 0 && (
                <CollapsibleSection title="Attributes">
                  <table className="text-sm border-collapse w-full max-w-sm">
                    <tbody>
                      {Object.entries(attributes).map(([k, v]) => (
                        <tr key={k} className="border-t">
                          <td className="py-1 pr-4 text-zinc-500">{k}</td>
                          <td className="py-1 break-all">{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CollapsibleSection>
              )}
              <CollapsibleSection title="System">
                <table className="text-sm border-collapse w-full max-w-sm">
                  <tbody>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Network</td>
                      <td className="py-1">{spec.network ? (spec.network === 'public' ? <AsDefault value="public" /> : spec.network) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Port</td>
                      <td className="py-1">{spec.port ?? <AsDefault value="none" />}</td>
                    </tr>
                    {spec.process.type === 'jvm' && (
                      <tr className="border-t">
                        <td className="py-1 pr-4 text-zinc-500">JVM profile</td>
                        <td className="py-1">{spec.process.profile != null ? String(spec.process.profile) : <Unset />}</td>
                      </tr>
                    )}
                    {instColl != null && (
                      <tr className="border-t">
                        <td className="py-1 pr-4 text-zinc-500">Instance collection</td>
                        <td className="py-1 font-mono">{instColl}</td>
                      </tr>
                    )}
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Service account</td>
                      <td className="py-1">{spec.service_account ? `${spec.service_account.namespace}/${spec.service_account.name}` : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Mount tokens</td>
                      <td className="py-1">{spec.mount_tokens != null ? String(spec.mount_tokens) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Unconfined</td>
                      <td className="py-1">{spec.unconfined != null ? String(spec.unconfined) : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Job group ID</td>
                      <td className="py-1">{spec.absolute_job_group_id ?? <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">Parent IDs</td>
                      <td className="py-1">{spec.absolute_parent_ids?.length ? spec.absolute_parent_ids.join(', ') : <Unset />}</td>
                    </tr>
                    <tr className="border-t">
                      <td className="py-1 pr-4 text-zinc-500">In-update parent IDs</td>
                      <td className="py-1">{spec.in_update_parent_ids?.length ? spec.in_update_parent_ids.join(', ') : <Unset />}</td>
                    </tr>
                  </tbody>
                </table>
                <div>
                  <h3 className="text-sm font-medium text-zinc-500 mb-1">Secrets</h3>
                  {spec.secrets && spec.secrets.length > 0 ? (
                    <table className="text-sm border-collapse w-full">
                      <thead><tr className="text-left text-zinc-500"><th className="py-1 pr-4">Secret</th><th className="py-1">Mount path</th></tr></thead>
                      <tbody>
                        {spec.secrets.map((s, i) => (
                          <tr key={i} className="border-t">
                            <td className="py-1 pr-4 font-mono text-xs">{s.namespace}/{s.name}</td>
                            <td className="py-1 font-mono text-xs">{s.mount_path}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="text-zinc-400 text-xs">None.</div>
                  )}
                </div>
              </CollapsibleSection>
            </>
          ) : (
            <div className="text-zinc-400 text-sm">No spec available.</div>
          )}
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
