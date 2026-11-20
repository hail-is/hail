import { describe, expect, it } from 'vitest';
import { labelEffect, sortLabels, visibleLabels } from './labelModel';

describe('visibleLabels', () => {
  it('keeps only labels CI acts on, most significant first', () => {
    expect(visibleLabels(['rerun all tests', 'docs', 'prio:high', 'WIP', 'do-not-test', 'stacked PR'])).toEqual([
      'stacked PR',
      'WIP',
      'prio:high',
      'do-not-test',
      'rerun all tests',
    ]);
  });

  it('sorts unknown labels last when keeping all of them', () => {
    expect(sortLabels(['docs', 'prio:high', 'bug', 'WIP'])).toEqual(['WIP', 'prio:high', 'bug', 'docs']);
  });

  it('is case-sensitive, matching CI', () => {
    expect(visibleLabels(['wip'])).toEqual([]);
    expect(labelEffect('wip')).toBeUndefined();
  });
});
