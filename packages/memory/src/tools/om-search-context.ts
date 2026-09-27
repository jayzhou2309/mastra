import { isDeepStrictEqual } from 'node:util';
import type { MastraDBMessage } from '@mastra/core/agent';
import { resolveToolResultValue } from '../processors/observational-memory/tool-result-helpers';
import type { RecallMemory, RecallSearchResult } from './om-tools';

export function searchContextKey(match: Pick<RecallSearchResult, 'threadId' | 'groupId'>): string {
  return JSON.stringify([match.threadId, match.groupId]);
}

/** Read only real recall results, not quoted tool output in user or assistant text. */
export function getVisibleSearchExcerpts(messages: readonly MastraDBMessage[]): Map<string, string[]> {
  const excerpts = new Map<string, string[]>();
  for (const message of messages) {
    if (message.role !== 'assistant') continue;
    for (const part of message.content.parts) {
      if (part.type !== 'tool-invocation') continue;
      const invocation = part.toolInvocation;
      if (invocation.toolName !== 'recall' || invocation.state !== 'result' || invocation.args?.mode !== 'search')
        continue;
      const resolved = resolveToolResultValue(part, invocation.result);
      let value = resolved.value;
      if (resolved.usingStoredModelOutput) {
        // A custom model output can hide the original result. Never dedupe against that hidden value.
        if (!value || typeof value !== 'object' || !('type' in value) || value.type !== 'json' || !('value' in value))
          continue;
        value = value.value;
      }
      if (!value || typeof value !== 'object' || !('results' in value) || typeof value.results !== 'string') continue;
      for (const section of value.results.split(/^### (?:Current thread memory|Memory from another thread)\s*$/m)) {
        const threadId = /^- thread: (\S+)/m.exec(section)?.[1];
        const groupId = /^- observation group: ([^\n]+)$/m.exec(section)?.[1];
        const excerpt = /\n```text\n([\s\S]*?)\n```/.exec(section)?.[1];
        if (!threadId || !groupId || !excerpt) continue;
        const key = searchContextKey({ threadId, groupId });
        const previous = excerpts.get(key) ?? [];
        previous.push(excerpt);
        excerpts.set(key, previous);
      }
    }
  }
  return excerpts;
}

function visibleParts(message: MastraDBMessage) {
  return message.content.parts.filter(part => !part.type.startsWith('data-') && part.type !== 'step-start');
}

export async function sourceRangeIsVisible({
  match,
  messages,
  memory,
  resourceId,
}: {
  match: RecallSearchResult;
  messages: readonly MastraDBMessage[];
  memory: Pick<RecallMemory, 'recall'>;
  resourceId: string;
}): Promise<boolean> {
  if (!match.groupId || !match.range) return false;
  const endpoints = /^([^:,]+):([^:,]+)$/.exec(match.range);
  if (!endpoints) return false;
  const byId = new Map(
    messages.filter(message => message.threadId === match.threadId).map(message => [message.id, message]),
  );
  const start = byId.get(endpoints[1]!);
  const end = byId.get(endpoints[2]!);
  if (!start || !end || start.createdAt > end.createdAt) return false;

  // Endpoints alone cannot prove coverage: processors may have removed messages in between.
  // Bound the read by the current context size; an incomplete window cannot justify suppression.
  const history = await memory.recall({
    threadId: match.threadId,
    resourceId,
    page: 0,
    perPage: byId.size + 1,
    orderBy: { field: 'createdAt', direction: 'ASC' },
    filter: { dateRange: { start: start.createdAt, end: end.createdAt } },
  });
  if (history.hasMore) return false;
  const first = history.messages.findIndex(message => message.id === start.id);
  const last = history.messages.findIndex(message => message.id === end.id);
  if (first < 0 || last < first) return false;
  return history.messages.slice(first, last + 1).every(source => {
    const current = byId.get(source.id);
    if (!current || current.role !== source.role || source.threadId !== match.threadId) return false;
    const sourceParts = visibleParts(source);
    const currentParts = visibleParts(current);
    if (!sourceParts.length) return false;
    let position = 0;
    return sourceParts.every(part => {
      const index = currentParts.findIndex(
        (candidate, index) => index >= position && isDeepStrictEqual(candidate, part),
      );
      position = index + 1;
      return index >= 0;
    });
  });
}
