import { MockLanguageModelV2 } from '@internal/ai-sdk-v5/test';
import { MessageList } from '@mastra/core/agent';
import { RequestContext } from '@mastra/core/request-context';
import { InMemoryDB, InMemoryMemory } from '@mastra/core/storage';
import { describe, expect, it, vi } from 'vitest';
import { getRetrievalInstructions } from '../constants';
import { renderObservationGroupsForReflection } from '../observation-groups';
import { ObservationalMemory } from '../observational-memory';
import { ObservationalMemoryProcessor } from '../processor';
import type { MemoryContextProvider } from '../processor';

describe('actor recall guidance', () => {
  it.each(['thread', 'resource'] as const)('teaches search-to-observation paging in %s scope', scope => {
    const text = getRetrievalInstructions(scope);
    expect(text).toContain('mode: "observations"');
    expect(text).toContain('5 groups');
    expect(text).toContain('two views of the same conversation history');
    expect(text).toContain('observations for breadth, and messages for depth');
    expect(text).toContain('page both before and after the anchor');
    expect(text).toContain('The range connects the summary view to the raw-message view');
    expect(text).toContain('limit: 2');
    expect(text).toContain('currently provided tool schema');
    expect(text).toContain('what was discussed or decided and why, start with recall');
    expect(text).toContain('Distinguish recorded reasons from your own inference');
    expect(text).toContain('hasMore: false');
    expect(text).toContain('not whether older raw messages exist');
    expect(text).toContain('first or last group ID');
    expect(text).toContain('direction: "before"');
    expect(text).toContain('direction: "after"');
    expect(text).toContain('not a complete timeline');
    expect(text).toContain('both event dates');
    expect(text).toContain('missing search hit is not evidence');
    expect(text).toContain('kind="reflection"');
    expect(text).toContain('lossy summaries');
    expect(text).toContain('that does not mean it did not happen');
    expect(text).toContain("user's message verbatim as the search query");
    expect(text).not.toContain('There is no relevant range in your observations for the topic');
    expect(text).not.toContain('go straight to `mode: "messages"`');
  });

  it.each(['thread', 'resource'] as const)('does not advertise unavailable paging in %s scope', async scope => {
    const browsing = getRetrievalInstructions(scope, undefined, false);
    expect(browsing).not.toContain('mode: "observations"');
    expect(browsing).not.toContain('mode: "search"');
    const storage = new InMemoryMemory({ db: new InMemoryDB() });
    Object.defineProperty(storage, 'supportsObservationalMemoryHistorySearch', { value: false });
    const om = new ObservationalMemory({ storage, model: 'test-model', retrieval: { scope, vector: true } });
    const text = (await om.buildContextSystemMessages({ threadId: 'thread', resourceId: 'resource' }))!.join('\n');
    expect(text).toContain('mode: "search"');
    expect(text).not.toContain('mode: "observations"');
  });

  it('labels lossy reflection groups in actor context without changing reflector input', async () => {
    const storage = new InMemoryMemory({ db: new InMemoryDB() });
    const record = await storage.initializeObservationalMemory({
      threadId: 'thread',
      resourceId: 'resource',
      scope: 'thread',
      config: {},
    });
    const observations =
      '<observation-group id="summary" range="m1:m2" kind="reflection">Broad summary</observation-group>\n<observation-group id="original" range="m3:m4">Original note</observation-group>';
    await storage.updateActiveObservations({ id: record.id, observations, tokenCount: 10, lastObservedAt: new Date() });
    const om = new ObservationalMemory({ storage, model: 'test-model', retrieval: { vector: true } });
    const text = await om.buildContextSystemMessage({ threadId: 'thread', resourceId: 'resource' });
    expect(text).toContain('## Group `summary`\n_kind: reflection_\n_range: `m1:m2`_');
    expect(text).toContain('## Group `original`\n_range: `m3:m4`_');
    expect(renderObservationGroupsForReflection(observations)).not.toContain('_kind: reflection_');
  });

  it.each([false, true])('omits recall guidance when disabled, with observations=%s', async populated => {
    const storage = new InMemoryMemory({ db: new InMemoryDB() });
    const record = await storage.initializeObservationalMemory({
      threadId: 'thread',
      resourceId: 'resource',
      scope: 'thread',
      config: {},
    });
    if (populated)
      await storage.updateActiveObservations({
        id: record.id,
        observations: 'A fact.',
        tokenCount: 3,
        lastObservedAt: new Date(),
      });
    const om = new ObservationalMemory({ storage, model: 'test-model' });
    const text = await om.buildContextSystemMessage({ threadId: 'thread', resourceId: 'resource' });
    expect(text ?? '').not.toContain('## Recall');
  });

  for (const scope of ['thread', 'resource'] as const) {
    for (const readOnly of [false, true]) {
      it.each(['missing', 'empty', 'populated'] as const)(
        `injects stable system guidance on every step: ${scope}, readOnly=${readOnly}, record=%s`,
        async recordState => {
          const threadId = 'thread';
          const resourceId = 'resource';
          const storage = new InMemoryMemory({ db: new InMemoryDB() });
          await storage.saveThread({
            thread: { id: threadId, resourceId, createdAt: new Date(), updatedAt: new Date() },
          });
          if (recordState !== 'missing') {
            const record = await storage.initializeObservationalMemory({
              threadId,
              resourceId,
              scope: 'thread',
              config: {},
            });
            if (recordState === 'populated') {
              await storage.updateActiveObservations({
                id: record.id,
                observations: 'Existing user preferences.',
                tokenCount: 5,
                lastObservedAt: new Date(),
              });
            }
          }
          const initialize = vi.spyOn(storage, 'initializeObservationalMemory');
          const om = new ObservationalMemory({
            storage,
            model: 'test-model',
            retrieval: { scope, vector: true, instructions: 'Custom application guidance.' },
            observation: { messageTokens: 100_000, bufferTokens: false },
            reflection: { observationTokens: 100_000, bufferActivation: 1 },
          });
          const memory: MemoryContextProvider = {
            getContext: async () => {
              const omRecord = await om.getRecord(threadId, resourceId);
              return {
                omRecord,
                hasObservations: !!omRecord?.activeObservations,
                messages: [],
                systemMessage: undefined,
                continuationMessage: undefined,
                otherThreadsContext: undefined,
              };
            },
            persistMessages: vi.fn(),
          };
          const processor = new ObservationalMemoryProcessor(om, memory);
          const messageList = new MessageList({ threadId, resourceId });
          const requestContext = new RequestContext();
          requestContext.set('MastraMemory', { thread: { id: threadId }, resourceId, memoryConfig: { readOnly } });
          const state = {};
          let firstPrefix: string | undefined;
          for (const stepNumber of [0, 1]) {
            await processor.processInputStep({
              messageList,
              messages: [],
              requestContext,
              stepNumber,
              state,
              steps: [],
              systemMessages: [],
              model: new MockLanguageModelV2() as any,
              retryCount: 0,
              abort: () => {
                throw new Error('Unexpected abort');
              },
            });
            const messages = messageList.getSystemMessages('observational-memory');
            expect(messages.every(message => message.role === 'system')).toBe(true);
            const text = messages.map(message => message.content).join('\n');
            expect(text.match(/## Recall — looking up source messages/g)).toHaveLength(1);
            expect(text).toContain('mode: "observations"');
            expect(text).toContain('both event dates');
            expect(text).toContain('Custom application guidance.');
            const prefix = String(messages[0]!.content);
            if (stepNumber === 0) firstPrefix = prefix;
            else expect(prefix).toBe(firstPrefix);
          }
          if (readOnly) {
            expect(initialize).not.toHaveBeenCalled();
            expect(memory.persistMessages).not.toHaveBeenCalled();
          }
        },
      );
    }
  }
});
