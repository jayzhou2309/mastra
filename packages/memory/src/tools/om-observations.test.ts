import { describe, expect, it, vi } from 'vitest';
import xxhash from 'xxhash-wasm';
import { wrapInObservationGroup } from '../processors/observational-memory/observation-groups';
import { findGroupTimeline, gapMarkerBetween, pageObservationGroups } from './om-observations';
import type { OMGenerationRecord, OMTimelineEngine } from './om-observations';

const group = (id: string, kind?: string) =>
  wrapInObservationGroup(`Date: Jan 1, 2024\n${id} original details`, `${id}-start:${id}-end`, id, undefined, kind);
const record = (generationCount: number, ids: string[]): OMGenerationRecord => ({
  id: `generation-${generationCount}`,
  generationCount,
  threadId: 'thread',
  observedTimezone: 'UTC',
  activeObservations: ids.map(id => group(id, id.startsWith('reflection') ? 'reflection' : undefined)).join('\n'),
});
function history(records: OMGenerationRecord[]) {
  const getHistory = vi.fn<OMTimelineEngine['getHistory']>(async (_thread, _resource, limit, options) =>
    records
      .filter(
        r =>
          options?.groupId === undefined ||
          r.activeObservations.includes(`<observation-group id="${options.groupId}"`) ||
          r.bufferedObservationChunks?.some(chunk =>
            chunk.observations.includes(`<observation-group id="${options.groupId}"`),
          ),
      )
      .filter(r => options?.beforeGeneration === undefined || r.generationCount < options.beforeGeneration)
      .filter(r => options?.afterGeneration === undefined || r.generationCount > options.afterGeneration)
      .sort((a, b) => (options?.sortDirection === 'ASC' ? 1 : -1) * (a.generationCount - b.generationCount))
      .slice(0, limit),
  );
  return { getHistory };
}
const chunk = (observations: string) => ({
  id: 'chunk',
  cycleId: 'cycle',
  observations,
  tokenCount: 1,
  messageIds: ['m1'],
  messageTokens: 1,
  lastObservedAt: new Date('2024-01-01'),
  createdAt: new Date('2024-01-01'),
});
const ids = (text: string) => [...text.matchAll(/^## Group `([^`]+)`/gm)].map(m => m[1]);
const args = { threadId: 'thread', resourceId: 'resource', limit: 3 };
const generations = () => [
  record(0, ['a', 'b', 'c', 'd']),
  record(1, ['reflection-1', 'c', 'd', 'e', 'f']),
  record(2, ['reflection-2', 'e', 'f', 'g', 'h']),
];

describe('observation group history', () => {
  it.each([
    ['c', undefined, ['c', 'd', 'e']],
    ['b', 'after', ['c', 'd', 'e']],
    ['d', 'before', ['a', 'b', 'c']],
    ['d', 'after', ['e', 'f']],
  ] as const)(
    'pages buffered anchor %s %s across active, buffered, and historical content',
    async (groupId, direction, expected) => {
      const records = [
        {
          ...record(0, ['a', 'b']),
          bufferedObservationChunks: [
            chunk(group('b') + group('c')),
            chunk(group('reflection-buffer', 'reflection') + group('d')),
          ],
        },
        { ...record(1, ['reflection-1', 'c', 'd', 'e']), bufferedObservationChunks: [chunk(group('f'))] },
      ];
      const before = structuredClone(records);
      const om = history(records);
      const page = await pageObservationGroups({ om, ...args, groupId, direction });
      expect(ids(page.results)).toEqual(expected);
      expect(page.results).not.toContain('reflection-buffer');
      expect(records).toEqual(before);
      expect(om.getHistory.mock.calls.every(call => call[2] === 1)).toBe(true);
    },
  );
  it('keeps the same group cursor usable after buffered content is activated', async () => {
    const head = { ...record(0, ['a']), bufferedObservationChunks: [chunk(group('b') + group('c'))] };
    const om = history([head]);
    expect(ids((await pageObservationGroups({ om, ...args, groupId: 'b', limit: 1 })).results)).toEqual(['b']);
    head.activeObservations += '\n' + head.bufferedObservationChunks[0]!.observations;
    head.bufferedObservationChunks = [];
    expect(ids((await pageObservationGroups({ om, ...args, groupId: 'b', direction: 'after' })).results)).toEqual([
      'c',
    ]);
  });
  it.each([false, true])('filters buffered resource-scoped groups by thread (hashed=%s)', async hashed => {
    const threadTag = hashed ? (await xxhash()).h32ToString('thread') : 'thread';
    const om = history([
      {
        ...record(0, []),
        threadId: null,
        bufferedObservationChunks: [
          chunk(
            `<thread id="other">${group('secret')}</thread><thread id="${threadTag}">${group('a')}${group('b')}</thread>`,
          ),
        ],
      },
    ]);
    expect(ids((await pageObservationGroups({ om, ...args, groupId: 'a' })).results)).toEqual(['a', 'b']);
    expect(await findGroupTimeline(om, 'thread', 'resource', 'secret')).toBeNull();
  });
  it('locates the earliest retained occurrence with one filtered record, independent of dates', async () => {
    const om = history([
      record(0, ['a']),
      ...Array.from({ length: 1001 }, (_, i) => record(i + 1, ['reflection', 'b'])),
    ]);
    const result = await findGroupTimeline(om, 'thread', 'resource', 'a');
    expect(result?.record.id).toBe('generation-0');
    expect(om.getHistory).toHaveBeenCalledExactlyOnceWith('thread', 'resource', 1, {
      groupId: 'a',
      sortDirection: 'ASC',
    });
  });
  it.each([
    ['c', 'after', ['d', 'e', 'f']],
    ['e', 'before', ['b', 'c', 'd']],
    ['h', 'before', ['e', 'f', 'g']],
    ['f', 'after', ['g', 'h']],
    ['a', 'before', []],
    ['h', 'after', []],
  ] as const)('pages %s %s without reflections or duplicates', async (groupId, direction, expected) => {
    const om = history(generations());
    const result = await pageObservationGroups({ om, ...args, groupId, direction });
    expect(ids(result.results)).toEqual(expected);
    expect(result.count).toBe(expected.length);
    expect(result.results).not.toContain('reflection-');
    expect(om.getHistory.mock.calls.every(call => call[2] === 1)).toBe(true);
  });
  it.each([
    ['a', undefined, 3, true],
    ['f', undefined, 3, false],
    ['e', 'after', 3, false],
    ['f', 'after', 3, false],
    ['e', 'before', 3, true],
    ['d', 'before', 3, false],
    ['b', 'before', 3, false],
    ['a', 'before', 3, false],
    ['h', 'after', 3, false],
  ] as const)('reports whether another page exists from %s %s', async (groupId, direction, limit, hasMore) => {
    const result = await pageObservationGroups({ om: history(generations()), ...args, groupId, direction, limit });
    expect(result.hasMore).toBe(hasMore);
    const forward = direction !== 'before';
    expect(result.results.includes(forward ? 'Browse later:' : 'Browse earlier:')).toBe(hasMore);
    if (!hasMore && result.count > 0) {
      expect(result.results).toContain(
        forward ? 'End of retained observation history' : 'Start of retained observation history',
      );
    }
  });
  it('does not advertise an earlier page when an inclusive page starts the history', async () => {
    const result = await pageObservationGroups({ om: history(generations()), ...args, groupId: 'a' });
    expect(result.results).toContain('Start of retained observation history');
    expect(result.results).not.toContain('Browse earlier:');
  });
  it.each([false, true])('checks earlier originals across reflection-only generations (earlier=%s)', async earlier => {
    const om = history([record(0, earlier ? ['a'] : ['reflection']), record(2, ['reflection']), record(4, ['b'])]);
    const result = await pageObservationGroups({ om, ...args, groupId: 'b', limit: 1 });
    expect(result.results.includes('Browse earlier:')).toBe(earlier);
    expect(result.results.includes('Start of retained observation history')).toBe(!earlier);
    expect(result.hasMore).toBe(false);
    expect(result.results).toContain('End of retained observation history');
    expect(om.getHistory.mock.calls.every(call => call[2] === 1)).toBe(true);
  });
  it('returns the full anchor and multiple groups when direction is omitted', async () => {
    const result = await pageObservationGroups({ om: history(generations()), ...args, groupId: 'd' });
    expect(ids(result.results)).toEqual(['d', 'e', 'f']);
    expect(result.results).toContain('_range: `d-start:d-end`_');
  });
  it('can walk forward and back across multiple pages without losing carried groups', async () => {
    const om = history(generations());
    const forward: string[] = [];
    let anchor = 'a';
    for (let i = 0; i < 5; i++) {
      const page = await pageObservationGroups({ om, ...args, limit: 2, groupId: anchor, direction: 'after' });
      const found = ids(page.results);
      expect(found.length).toBeGreaterThan(0);
      forward.push(...found);
      anchor = found.at(-1)!;
      if (!page.hasMore) break;
    }
    expect(forward).toEqual(['b', 'c', 'd', 'e', 'f', 'g', 'h']);
    const backward: string[] = [];
    for (let i = 0; i < 5; i++) {
      const page = await pageObservationGroups({ om, ...args, limit: 2, groupId: anchor, direction: 'before' });
      const found = ids(page.results);
      expect(found.length).toBeGreaterThan(0);
      backward.unshift(...found);
      anchor = found[0]!;
      if (!page.hasMore) break;
    }
    expect(backward).toEqual(['a', 'b', 'c', 'd', 'e', 'f', 'g']);
  });
  it('skips reflection-only generations and handles gaps in retained generation numbers', async () => {
    const om = history([record(0, ['a']), record(3, ['reflection-3']), record(7, ['b', 'c'])]);
    expect(ids((await pageObservationGroups({ om, ...args, groupId: 'c', direction: 'before' })).results)).toEqual([
      'a',
      'b',
    ]);
    expect(await findGroupTimeline(om, 'thread', 'resource', 'reflection-3')).toBeNull();
  });
  it('returns no body for missing groups or a mismatched thread', async () => {
    const om = history(generations());
    expect(await findGroupTimeline(om, 'other', 'resource', 'a')).toBeNull();
    const missing = await pageObservationGroups({ om, ...args, groupId: 'missing' });
    expect(missing.count).toBe(0);
    expect(missing).not.toHaveProperty('hasMore');
  });
  it.each([false, true])('keeps resource-scoped paging within attributed thread (hashed=%s)', async hashed => {
    const threadTag = hashed ? (await xxhash()).h32ToString('thread') : 'thread';
    const om = history([
      {
        ...record(0, []),
        threadId: null,
        activeObservations: `<thread id="other">${group('secret')}</thread>\n<thread id="${threadTag}">${group('a')}${group('b')}</thread>`,
      },
    ]);
    const page = await pageObservationGroups({ om, ...args, groupId: 'a' });
    expect(ids(page.results)).toEqual(['a', 'b']);
    expect(page.results).not.toContain('secret');
    expect(await findGroupTimeline(om, 'thread', 'resource', 'secret')).toBeNull();
  });
  it('counts exact local gaps and conservative cross-generation gaps without duplicate carryover', async () => {
    const om = history(generations());
    const locate = (id: string) => findGroupTimeline(om, 'thread', 'resource', id);
    const hit = (groupId: string) => ({ threadId: 'thread', groupId });
    expect(gapMarkerBetween(hit('a'), hit('d'), await locate('a'), await locate('d'))).toContain(
      '2 observation groups hidden',
    );
    expect(gapMarkerBetween(hit('b'), hit('f'), await locate('b'), await locate('f'))).toContain(
      '3+ observation groups hidden',
    );
    expect(gapMarkerBetween(hit('c'), hit('f'), await locate('c'), await locate('f'))).toContain(
      '2+ observation groups hidden',
    );
    expect(
      gapMarkerBetween(hit('a'), { threadId: 'other', groupId: 'd' }, await locate('a'), await locate('d')),
    ).toBeNull();
  });
  it('renders dates in each record timezone', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-02T01:00:00Z'));
    try {
      const om = history([{ ...record(0, ['a']), observedTimezone: 'America/Los_Angeles' }]);
      const page = await pageObservationGroups({ om, ...args, groupId: 'a' });
      expect(page.results).toContain('Date: Jan 1, 2024 (today)');
    } finally {
      vi.useRealTimers();
    }
  });
});
