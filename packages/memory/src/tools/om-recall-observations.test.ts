import { standardSchemaToJSONSchema } from '@mastra/core/schema';
import { InMemoryStore } from '@mastra/core/storage';
import { estimateTokenCount } from 'tokenx';
import { describe, expect, it, vi } from 'vitest';
import { Memory } from '../index';
import { wrapInObservationGroup } from '../processors/observational-memory/observation-groups';
import type { OMTimelineEngine } from './om-observations';
import { recallTool, searchMessagesForResource } from './om-tools';

type RecallMemory = Parameters<typeof searchMessagesForResource>[0]['memory'];
const date = new Date('2024-01-01T12:00:00Z');
const thread = { id: 'thread', resourceId: 'resource', title: 'History', createdAt: date, updatedAt: date };
const group = (id: string) => wrapInObservationGroup(`Date: Jan 1, 2024\n${id} details`, `${id}-start:${id}-end`, id);
function setup() {
  const memory: RecallMemory = {
    getMemoryStore: async () => ({ listMessagesById: async () => ({ messages: [] }) }),
    recall: async () => ({ messages: [], total: 0, page: 0, perPage: 20, hasMore: false }),
    listThreads: async () => ({ threads: [thread], total: 1, hasMore: false, page: 0 }),
    getThreadById: vi.fn(async ({ threadId }) => (threadId === thread.id ? thread : null)),
    searchMessages: vi.fn(async () => ({
      results: [
        {
          threadId: 'thread',
          groupId: 'c',
          score: 0.99,
          observedAt: new Date('2024-01-03T12:00:00Z'),
          text: 'Date: Jan 3, 2024\nC',
        },
        { threadId: 'thread', groupId: 'a', score: 0.8, observedAt: date, text: 'Date: Jan 1, 2024\nA' },
        { threadId: 'thread', groupId: 'b', score: 0.2, observedAt: new Date('2024-01-02T12:00:00Z'), text: 'B' },
      ],
    })),
  };
  const om: OMTimelineEngine = {
    getHistory: vi.fn(async (_thread, _resource, _limit, options) => {
      if (options?.beforeGeneration !== undefined || options?.afterGeneration !== undefined) return [];
      return [
        {
          id: 'record',
          threadId: 'thread',
          generationCount: 0,
          observedTimezone: 'UTC',
          activeObservations: ['a', 'b', 'c'].map(group).join('\n'),
        },
      ];
    }),
  };
  return { memory, om };
}

describe('recall observations integration', () => {
  it('selects by similarity, renders chronologically with dates, and reports skipped groups', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-04T01:00:00Z'));
    try {
      const { memory, om } = setup();
      const result = await searchMessagesForResource({ memory, om, resourceId: 'resource', query: 'topic', topK: 2 });
      expect(result.count).toBe(2);
      expect(result.results.indexOf('observation group: a')).toBeLessThan(
        result.results.indexOf('observation group: c'),
      );
      expect(result.results).not.toContain('observation group: b');
      expect(result.results).toContain('1 observation groups hidden');
      expect(result.results).toContain('observed: 2024-01-01 12:00:00Z (3 days ago)');
      expect(result.results).toContain('Date: Jan 1, 2024 (3 days ago)');
      expect(result.results).not.toContain('thread updated');
      expect(om.getHistory).toHaveBeenCalledTimes(2);
      expect(om.getHistory).toHaveBeenCalledWith('thread', 'resource', 1, { groupId: 'a', sortDirection: 'ASC' });
    } finally {
      vi.useRealTimers();
    }
  });
  it('keeps undated hits after dated hits and surfaces truncation with recovery guidance', async () => {
    const { memory, om } = setup();
    memory.searchMessages = async () => ({
      results: [
        { threadId: 'thread', groupId: 'unknown', score: 1, text: 'undated' },
        { threadId: 'thread', groupId: 'dated', score: 0.5, observedAt: date, text: 'many words '.repeat(1000) },
      ],
    });
    const full = await searchMessagesForResource({ memory, resourceId: 'resource', query: 'x', maxTokens: 10000 });
    expect(full.results.indexOf('observation group: dated')).toBeLessThan(
      full.results.indexOf('observation group: unknown'),
    );
    const limited = await searchMessagesForResource({ memory, om, resourceId: 'resource', query: 'x', maxTokens: 150 });
    expect(limited.results).toContain('All 2 selected matching groups are shown; some excerpts are truncated');
    expect(limited.results).toContain('omitting direction');
    expect(limited.results).toContain('observation group: unknown');
    expect(limited.results).not.toContain('Narrow the query');
  });
  it('shares the text budget across hits and gives unused space to higher ranks before chronological rendering', async () => {
    const { memory, om } = setup();
    memory.searchMessages = async () => ({
      results: [
        {
          threadId: 'thread',
          groupId: 'c',
          score: 0.99,
          observedAt: new Date('2024-01-03'),
          text: 'newest '.repeat(1000),
        },
        { threadId: 'thread', groupId: 'a', score: 0.8, observedAt: date, text: 'oldest '.repeat(1000) },
        { threadId: 'thread', groupId: 'b', score: 0.2, observedAt: new Date('2024-01-02'), text: 'short' },
      ],
    });
    const result = await searchMessagesForResource({ memory, om, resourceId: 'resource', query: 'x', maxTokens: 300 });
    const excerpts = [...result.results.matchAll(/```text\n([\s\S]*?)^```/gm)].map(match => match[1].trimEnd());
    const tokens = excerpts.map(estimateTokenCount);
    expect(result.count).toBe(3);
    expect(excerpts).toHaveLength(3);
    expect(excerpts[0]).toMatch(/^oldest/);
    expect(excerpts[1]).toBe('short');
    expect(excerpts[2]).toMatch(/^newest/);
    expect(tokens[0]).toBeLessThanOrEqual(100);
    expect(tokens[2]).toBeGreaterThan(100);
    expect(tokens.reduce((sum, count) => sum + count, 0)).toBeLessThanOrEqual(300);
    expect(estimateTokenCount(result.results)).toBeGreaterThan(300);
    expect(result.results.match(/\[Excerpt truncated\]/g)).toHaveLength(2);
  });
  it('keeps every selected hit visible at the maximum result count', async () => {
    const { memory } = setup();
    memory.searchMessages = async () => ({
      results: Array.from({ length: 20 }, (_, index) => ({
        threadId: 'thread',
        groupId: `group-${index}`,
        score: 1 - index / 20,
        observedAt: new Date(Date.UTC(2024, 0, 20 - index)),
        text: `hit${index} `.repeat(1000),
      })),
    });
    const result = await searchMessagesForResource({ memory, resourceId: 'resource', query: 'x', topK: 20 });
    const excerpts = [...result.results.matchAll(/```text\n([\s\S]*?)^```/gm)].map(match => match[1].trimEnd());
    expect(result.count).toBe(20);
    expect(excerpts).toHaveLength(20);
    expect(excerpts.every(text => text.length > 0)).toBe(true);
    expect(excerpts[0]).toMatch(/^hit19 /);
    expect(excerpts[19]).toMatch(/^hit0 /);
    expect(excerpts.reduce((sum, text) => sum + estimateTokenCount(text), 0)).toBeLessThanOrEqual(2000);
    expect(result.results).toContain('All 20 selected matching groups are shown');
    expect(result.results).toContain('mode="messages"');
    expect(result.results).not.toContain('mode="observations"');
  });
  it('preserves complete gap navigation when the observation excerpts are shortened', async () => {
    const { memory, om } = setup();
    const search = memory.searchMessages!;
    memory.searchMessages = async input => ({
      results: (await search(input)).results.map(hit => ({ ...hit, text: 'dense observation '.repeat(1000) })),
    });
    const result = await searchMessagesForResource({
      memory,
      om,
      resourceId: 'resource',
      query: 'x',
      topK: 2,
      maxTokens: 40,
    });
    expect(result.results).toContain('observation group: a');
    expect(result.results).toContain('observation group: c');
    expect(result.results).toContain(
      '1 observation groups hidden between these results; continue with recall({"mode":"observations","threadId":"thread","groupId":"a","direction":"after"})',
    );
  });
  it('does not add truncation guidance when every hit fits', async () => {
    const { memory } = setup();
    const result = await searchMessagesForResource({ memory, resourceId: 'resource', query: 'x' });
    expect(result.count).toBe(3);
    expect(result.results).not.toContain('truncated');
  });
  it('exposes observation paging and date filters in both retrieval scopes', () => {
    for (const retrievalScope of ['resource', 'thread'] as const) {
      const tool = recallTool(undefined, { retrievalScope });
      const schema = standardSchemaToJSONSchema(tool.inputSchema) as {
        properties: Record<string, { enum?: string[]; description?: string }>;
      };
      expect(schema.properties.mode.enum).toContain('observations');
      expect(schema.properties.before.description).toContain('search');
      expect(schema.properties.after.description).toContain('search');
      expect(schema.properties.direction.description).toContain('full anchor');
    }
  });
  it('routes paging through the tool with multiple complete groups and correct continuation scope', async () => {
    const { memory, om } = setup();
    const tool = recallTool(undefined, { getOMEngine: () => om });
    const result = (await tool.execute?.({ mode: 'observations', groupId: 'a', limit: 2 }, {
      memory,
      agent: { threadId: 'thread', resourceId: 'resource' },
    } as any)) as any;
    expect(result.count).toBe(2);
    expect(result.hasMore).toBe(true);
    expect(result.results).toContain('Thread: "History"');
    expect(result.results).toContain('Showing 2 groups starting at `a` (oldest first)');
    expect(result.results).toContain('## Group `a`');
    expect(result.results).toContain('## Group `b`');
    expect(result.results).not.toContain('"threadId"');
    expect(result.results).toContain('"groupId":"b","direction":"after"');
  });
  it('rejects a cross-resource thread before fetching observations', async () => {
    const { memory, om } = setup();
    const tool = recallTool(undefined, { retrievalScope: 'resource', getOMEngine: () => om });
    await expect(
      tool.execute?.({ mode: 'observations', threadId: 'thread', groupId: 'a' }, {
        memory,
        agent: { threadId: 'other', resourceId: 'other-resource' },
      } as any),
    ).rejects.toThrow('Thread not found');
    expect(om.getHistory).not.toHaveBeenCalled();
  });
  it('does not bypass thread scope with an explicit sibling thread', async () => {
    const { memory, om } = setup();
    const tool = recallTool(undefined, { retrievalScope: 'thread', getOMEngine: () => om });
    await expect(
      tool.execute?.({ mode: 'observations', threadId: 'thread', groupId: 'a' }, {
        memory,
        agent: { threadId: 'other', resourceId: 'resource' },
      } as any),
    ).rejects.toThrow('Thread not found');
    expect(om.getHistory).not.toHaveBeenCalled();
  });
  it('fails closed on old adapters rather than fetching unfiltered history', async () => {
    const storage = new InMemoryStore();
    const store = (await storage.getStore('memory'))!;
    vi.spyOn(store, 'getObservationalMemoryHistory');
    Object.defineProperty(store, 'supportsObservationalMemoryHistorySearch', { value: false });
    const memory = new Memory({
      storage,
      vector: {} as any,
      embedder: {} as any,
      options: { observationalMemory: { model: 'test-model', retrieval: { vector: true } } },
    });
    const result = (await memory.listTools().recall.execute?.({ mode: 'observations', groupId: 'a' }, {
      memory,
      agent: { threadId: 'thread', resourceId: 'resource' },
    } as any)) as any;
    expect(result.count).toBe(0);
    expect(result.results).toContain('storage adapter');
    expect(store.getObservationalMemoryHistory).not.toHaveBeenCalled();
  });
});
