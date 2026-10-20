import { describe, expect, it } from 'vitest';
import { formatElapsed } from './timeUtils';

describe('formatElapsed', () => {
  it.each([
    [0, '0s'],
    [0.5, '0.5s'],
    [45, '45s'],
    [60, '1m'],
    [125, '2m 5s'],
    [3600, '1h'],
    [5400, '1h 30m'],
    [5425, '1h 30m'],
  ])('%d -> %s', (input, expected) => {
    expect(formatElapsed(input)).toBe(expected);
  });
});
