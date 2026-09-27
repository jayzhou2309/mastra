import type { MastraDBMessage } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { estimateTokenCount } from 'tokenx';
import { describe, expect, it, vi } from 'vitest';
import { Memory } from '../index';
import { recallTool, searchMessagesForResource } from './om-tools';
import type { RecallMemory, RecallSearchResult } from './om-tools';

const threadId = 'thread';
const resourceId = 'resource';
function message(id: string, seconds = 0): MastraDBMessage {
  return {
    id,
    threadId,
    resourceId,
    role: 'user',
    createdAt: new Date(1700000000000 + seconds * 1000),
    content: { format: 2, parts: [{ type: 'text', text: `Message ${id}` }] },
  };
}
function resultMessage(result: unknown): MastraDBMessage {
  return {
    ...message('recall-result'),
    role: 'assistant',
    content: {
      format: 2,
      parts: [
        {
          type: 'tool-invocation',
          toolInvocation: {
            toolCallId: 'previous-search',
            toolName: 'recall',
            state: 'result',
            args: { mode: 'search', query: 'earlier' },
            result,
          },
        },
      ],
    },
  };
}
function setup(hits: RecallSearchResult[], source: MastraDBMessage[] = []) {
  const memory: RecallMemory = {
    getMemoryStore: async () => ({ listMessagesById: async () => ({ messages: source }) }),
    recall: vi.fn(async () => ({
      messages: source,
      total: source.length,
      page: 0,
      perPage: source.length + 1,
      hasMore: false,
    })),
    listThreads: async () => ({ threads: [], total: 0, hasMore: false, page: 0 }),
    searchMessages: vi.fn(async () => ({ results: hits })),
  };
  const search = (currentMessages: MastraDBMessage[] = [], maxTokens = 100) =>
    searchMessagesForResource({ memory, resourceId, query: 'topic', currentMessages, maxTokens });
  return { memory, search };
}
const hit = (groupId: string, text: string, range?: string): RecallSearchResult => ({
  threadId,
  groupId,
  text,
  range,
  score: 1,
});
const excerpts = (text: string) => [...text.matchAll(/```text\n([\s\S]*?)\n```/g)].map(match => match[1]!);

describe('execution-time recall search context', () => {
  it('compacts identical excerpts and spends the reclaimed allowance on fresh hits', async () => {
    const a = hit('a', 'previous evidence '.repeat(500));
    const b = hit('b', 'new evidence '.repeat(500));
    const first = setup([a]);
    const previous = await first.search([], 50);
    const { search } = setup([a, b]);
    const full = await search();
    const compact = await search([resultMessage(previous)]);
    expect(compact.count).toBe(2);
    expect(compact.results).toContain('observation group: a\n  thread: thread; Excerpt already in current context.');
    expect(excerpts(compact.results)).toHaveLength(1);
    expect(estimateTokenCount(excerpts(compact.results)[0]!)).toBeGreaterThan(
      estimateTokenCount(excerpts(full.results)[1]!),
    );
    expect(excerpts(compact.results).reduce((sum, text) => sum + estimateTokenCount(text), 0)).toBeLessThanOrEqual(100);
  });

  it('returns compact references for all repeats, then expands again after the original results leave context', async () => {
    const { search } = setup([hit('a', 'Same evidence')]);
    const initial = await search();
    const compact = await search([resultMessage(initial)]);
    expect(compact.count).toBe(1);
    expect(excerpts(compact.results)).toEqual([]);
    expect(compact.results).not.toContain('truncated');
    expect((await search([resultMessage(compact)])).results).toBe(initial.results);
    expect((await search()).results).toBe(initial.results);
  });

  it.each(['longer', 'different', 'other-thread'] as const)(
    'keeps %s evidence for a previously seen group ID',
    async kind => {
      const { search } = setup([hit('a', 'first fragment '.repeat(300))]);
      const previous = await search([], 30);
      const next = hit('a', kind === 'different' ? 'another chunk of this group' : 'first fragment '.repeat(300));
      if (kind === 'other-thread') next.threadId = 'sibling';
      const result = await setup([next]).search([resultMessage(previous)], 100);
      expect(result.results).not.toContain('already in current context');
      expect(excerpts(result.results)).toHaveLength(1);
    },
  );

  it.each(['quoted', 'other-tool', 'custom-output'] as const)(
    'does not suppress against %s search-looking text',
    async kind => {
      const { search } = setup([hit('a', 'evidence')]);
      const previous = await search();
      const current = resultMessage(previous);
      const part = current.content.parts[0]!;
      if (kind === 'quoted') current.content.parts = [{ type: 'text', text: previous.results }];
      if (part.type === 'tool-invocation') {
        if (kind === 'other-tool') part.toolInvocation.toolName = 'other';
        if (kind === 'custom-output')
          part.providerMetadata = { mastra: { modelOutput: { type: 'text', value: 'hidden' } } };
      }
      expect((await search([current])).results).toBe(previous.results);
    },
  );

  it('recognizes a JSON model output that still contains the excerpt', async () => {
    const { search } = setup([hit('a', 'evidence')]);
    const previous = await search();
    const current = resultMessage({ results: 'not the model output' });
    current.content.parts[0]!.providerMetadata = { mastra: { modelOutput: { type: 'json', value: previous } } };
    expect((await search([current])).results).toContain('Excerpt already in current context');
  });

  it('suppresses a fully visible source range using a bounded read without modifying the messages', async () => {
    const source = [message('start'), message('middle', 1), message('end', 2)];
    const { search, memory } = setup([hit('a', 'summary', 'start:end')], source);
    const original = structuredClone(source);
    const result = await search(source);
    expect(result.results).toContain('Source messages already in current context.');
    expect(excerpts(result.results)).toEqual([]);
    expect(memory.recall).toHaveBeenCalledExactlyOnceWith({
      threadId,
      resourceId,
      page: 0,
      perPage: 4,
      orderBy: { field: 'createdAt', direction: 'ASC' },
      filter: { dateRange: { start: source[0]!.createdAt, end: source[2]!.createdAt } },
    });
    expect(source).toEqual(original);
    expect((await search()).results).toContain('summary');
  });

  it.each([
    'missing-middle',
    'trimmed-parts',
    'wrong-thread',
    'missing-end',
    'unpersisted',
    'incomplete-window',
    'bad-range',
  ] as const)('keeps an excerpt when source coverage is uncertain: %s', async kind => {
    const source = [message('start'), message('middle', 1), message('end', 2)];
    const current = structuredClone(source);
    if (kind === 'missing-middle') current.splice(1, 1);
    if (kind === 'trimmed-parts') current[1]!.content.parts = [{ type: 'text', text: 'trimmed' }];
    if (kind === 'wrong-thread')
      current.forEach(message => {
        message.threadId = 'other';
      });
    if (kind === 'missing-end') current.pop();
    const { search, memory } = setup(
      [hit('a', 'summary', kind === 'bad-range' ? 'start:end:extra' : 'start:end')],
      kind === 'unpersisted' ? [] : source,
    );
    if (kind === 'incomplete-window')
      memory.recall = vi.fn(async () => ({ messages: source, total: 100, page: 0, perPage: 4, hasMore: true }));
    expect((await search(current)).results).toContain('```text\nsummary\n```');
    if (['wrong-thread', 'missing-end', 'bad-range'].includes(kind)) expect(memory.recall).not.toHaveBeenCalled();
  });

  it('verifies source coverage through real memory storage, including a missing middle message', async () => {
    const memory = new Memory({ storage: new InMemoryStore() });
    const source = [message('start'), message('middle', 1), message('end', 2)];
    await memory.saveThread({
      thread: {
        id: threadId,
        resourceId,
        title: 'Source coverage',
        createdAt: source[0]!.createdAt,
        updatedAt: source[2]!.createdAt,
      },
    });
    await memory.saveMessages({ messages: source });
    vi.spyOn(memory, 'searchMessages').mockResolvedValue({ results: [hit('a', 'summary', 'start:end')] });
    const complete = await searchMessagesForResource({ memory, resourceId, query: 'topic', currentMessages: source });
    expect(complete.results).toContain('Source messages already in current context');
    const partial = await searchMessagesForResource({
      memory,
      resourceId,
      query: 'topic',
      currentMessages: [source[0]!, source[2]!],
    });
    expect(partial.results).toContain('```text\nsummary\n```');
  });

  it('handles a single-message range with later appended parts and non-visible OM markers', async () => {
    const source = [message('one')];
    const current = structuredClone(source);
    current[0]!.content.parts.push({ type: 'text', text: 'More recent content' });
    source[0]!.content.parts.push({ type: 'data-om-observation', data: { observed: true } });
    const { search } = setup([hit('a', 'summary', 'one:one')], source);
    expect((await search(current)).results).toContain('Source messages already in current context');
  });

  it('routes the live getter through recall.execute without using the input-only messages field', async () => {
    const { search, memory } = setup([hit('a', 'evidence')]);
    const previous = await search();
    const getMessages = vi.fn(() => [resultMessage(previous)]);
    const context = {
      memory,
      agent: {
        agentId: 'agent',
        toolCallId: 'call',
        threadId,
        resourceId,
        messages: [],
        getMessages,
        suspend: vi.fn(),
      },
    };
    const tool = recallTool();
    const result = await tool.execute?.({ mode: 'search', query: 'topic' }, context);
    expect(result).toEqual(
      expect.objectContaining({ results: expect.stringContaining('Excerpt already in current context') }),
    );
    expect(getMessages).toHaveBeenCalledTimes(1);
    getMessages.mockReturnValue([]);
    const next = await tool.execute?.({ mode: 'search', query: 'topic' }, context);
    expect(next).toEqual(expect.objectContaining({ results: expect.stringContaining('```text\nevidence\n```') }));
  });
});
