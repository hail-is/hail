import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { JobStatusPanel } from './JobStatusPanel';
import type { Job } from './jobModels';

const RETRIED_LABEL = 'Cost of retried attempts';
const RETRIED_CAVEAT = 'Attempts whose VM went away before they finished, most often because it was preempted.';
const PROJECTED_LABEL = 'Projected cost on non-preemptible';

function renderPanel(job: Partial<Job>) {
  render(
    <JobStatusPanel
      batchId="1"
      jobId="1"
      basePath="/batch"
      job={{ id: 1, batch_id: 1, state: 'Success', cost: 1.5, ...job }}
      latestAttempt={null}
      autoRefresh={false}
      isTerminal
      hasJvmProfile={false}
      onAutoRefreshToggle={() => {}}
      countdownKey={0}
      refreshIntervalMs={1000}
      jobRefreshing={false}
    />,
  );
}

afterEach(() => {
  cleanup();
});

describe('JobStatusPanel preemption costs', () => {
  it('shows the retried cost with its caveat and the projected cost', () => {
    renderPanel({ retried_attempts_cost: 0.25, projected_nonpreemptible_cost: 4.5 });
    expect(screen.getByText(RETRIED_LABEL).parentElement?.textContent).toContain('$0.25');
    expect(screen.queryByText(RETRIED_CAVEAT)).not.toBeNull();
    expect(screen.getByText(PROJECTED_LABEL).parentElement?.textContent).toContain('$4.50');
  });

  it('hides the retried cost when it is zero', () => {
    renderPanel({ retried_attempts_cost: 0, projected_nonpreemptible_cost: null });
    expect(screen.queryByText(RETRIED_LABEL)).toBeNull();
    expect(screen.queryByText(RETRIED_CAVEAT)).toBeNull();
  });

  it('hides both figures when they are null', () => {
    renderPanel({ retried_attempts_cost: null, projected_nonpreemptible_cost: null });
    expect(screen.queryByText(RETRIED_LABEL)).toBeNull();
    expect(screen.queryByText(PROJECTED_LABEL)).toBeNull();
  });
});
