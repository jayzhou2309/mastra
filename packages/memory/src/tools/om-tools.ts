import type { MastraDBMessage } from '@mastra/core/agent';
import type { MemoryConfigInternal } from '@mastra/core/memory';
import { createTool } from '@mastra/core/tools';
import type { JSONSchema7 } from 'json-schema';
import { estimateTokenCount } from 'tokenx';

import {
  addRelativeTimeToObservations,
  formatRelativeTime,
  resolveTimeZone,
} from '../processors/observational-memory/date-utils';
import { safeSlice } from '../processors/observational-memory/string-utils';
import {
  formatToolResultForObserver,
  resolveToolResultValue,
  truncateStringByTokens,
} from '../processors/observational-memory/tool-result-helpers';
import { findGroupTimeline, gapMarkerBetween, pageObservationGroups, pagingCall } from './om-observations';
import type { OMTimelineEngine } from './om-observations';
import { getVisibleSearchExcerpts, searchContextKey, sourceRangeIsVisible } from './om-search-context';

export type RecallDetail = 'low' | 'high';

function getMessageParts(msg: MastraDBMessage): any[] {
  if (typeof msg.content === 'string') return [];
  if (Array.isArray(msg.content)) return msg.content;
  const parts = msg.content?.parts;
  return Array.isArray(parts) ? parts : [];
}

/** Returns true if a message has at least one non-data part with visible content. */
function hasVisibleParts(msg: MastraDBMessage): boolean {
  if (typeof msg.content === 'string') return (msg.content as string).length > 0;
  const parts = getMessageParts(msg);
  if (parts.length === 0) return Boolean(msg.content?.content);
  return parts.some((p: { type?: string }) => !p.type?.startsWith('data-'));
}

type RecallThread = {
  id: string;
  title?: string;
  resourceId: string;
  createdAt: Date;
  updatedAt: Date;
};

export type RecallSearchResult = {
  threadId: string;
  score: number;
  groupId?: string;
  range?: string;
  text?: string;
  observedAt?: Date;
};

export type RecallMemory = {
  getMemoryStore: () => Promise<{
    listMessagesById: (args: { messageIds: string[] }) => Promise<{ messages: MastraDBMessage[] }>;
  }>;
  recall: (args: {
    threadId: string;
    resourceId?: string;
    page: number;
    perPage: number | false;
    orderBy?: { field: 'createdAt'; direction: 'ASC' | 'DESC' };
    filter?: {
      dateRange?: {
        start?: Date;
        end?: Date;
        startExclusive?: boolean;
        endExclusive?: boolean;
      };
    };
  }) => Promise<{
    messages: MastraDBMessage[];
    total: number;
    page: number;
    perPage: number | false;
    hasMore: boolean;
  }>;
  listThreads: (args: {
    perPage?: number | false;
    page?: number;
    orderBy?: { field: string; direction: 'ASC' | 'DESC' };
    filter?: { resourceId?: string; metadata?: Record<string, unknown> };
  }) => Promise<{ threads: RecallThread[]; total: number; hasMore: boolean; page: number }>;
  searchMessages?: (args: {
    query: string;
    resourceId: string;
    topK?: number;
    filter?: {
      threadId?: string;
      observedAfter?: Date;
      observedBefore?: Date;
    };
  }) => Promise<{ results: RecallSearchResult[] }>;
  getThreadById?: (args: { threadId: string }) => Promise<RecallThread | null>;
};

function parseRangeFormat(cursor: string): { startId: string; endId: string } | null {
  // Comma-separated merged ranges: "id1:id2,id3:id4"
  if (cursor.includes(',')) {
    const parts = cursor
      .split(',')
      .map(p => p.trim())
      .filter(Boolean);
    if (parts.length >= 1) {
      const first = parts[0]!;
      const last = parts[parts.length - 1]!;
      const firstColon = first.indexOf(':');
      const lastColon = last.indexOf(':');
      return {
        startId: firstColon > 0 ? first.slice(0, firstColon) : first,
        endId: lastColon > 0 ? last.slice(lastColon + 1) : last,
      };
    }
  }

  // Colon-delimited range: "startId:endId"
  const colonIndex = cursor.indexOf(':');
  if (colonIndex > 0 && colonIndex < cursor.length - 1) {
    return { startId: cursor.slice(0, colonIndex), endId: cursor.slice(colonIndex + 1) };
  }

  return null;
}

async function resolveCursorMessage(
  memory: RecallMemory,
  cursor: string,
  access?: { resourceId?: string; threadScope?: string; enforceThreadScope?: boolean },
): Promise<MastraDBMessage | { hint: string; startId: string; endId: string }> {
  const normalized = cursor.trim();

  if (!normalized) {
    throw new Error('Cursor is required');
  }

  const rangeIds = parseRangeFormat(normalized);
  if (rangeIds) {
    return {
      hint: `The cursor "${cursor}" looks like a range. Use one of the individual message IDs as the cursor instead: start="${rangeIds.startId}" or end="${rangeIds.endId}".`,
      ...rangeIds,
    };
  }

  const memoryStore = await memory.getMemoryStore();
  const result = await memoryStore.listMessagesById({ messageIds: [normalized] });
  let message = result.messages.find(message => message.id === normalized) ?? null;

  if (!message) {
    message = await resolveCursorMessageByRecall(memory, normalized, access);
  }

  if (!message) {
    throw new Error(`Could not resolve cursor message: ${cursor}`);
  }

  // Verify the cursor message belongs to the current resource
  if (access?.resourceId && message.resourceId !== access.resourceId) {
    throw new Error(`Could not resolve cursor message: ${cursor}`);
  }

  // In strict thread scope, verify the cursor belongs to the current thread
  if (access?.enforceThreadScope && access.threadScope && message.threadId !== access.threadScope) {
    throw new Error(`Could not resolve cursor message: ${cursor}`);
  }

  return message;
}

async function resolveCursorMessageByRecall(
  memory: RecallMemory,
  cursor: string,
  access?: { resourceId?: string; threadScope?: string; enforceThreadScope?: boolean },
): Promise<MastraDBMessage | null> {
  if (access?.enforceThreadScope && access.threadScope) {
    const result = await memory.recall({
      threadId: access.threadScope,
      resourceId: access.resourceId,
      page: 0,
      perPage: false,
    });

    return result.messages.find(message => message.id === cursor) ?? null;
  }

  if (!access?.resourceId) {
    return null;
  }

  const threads = await memory.listThreads({
    page: 0,
    perPage: 100,
    orderBy: { field: 'updatedAt', direction: 'DESC' },
    filter: { resourceId: access.resourceId },
  });

  for (const thread of threads.threads) {
    const result = await memory.recall({
      threadId: thread.id,
      resourceId: access.resourceId,
      page: 0,
      perPage: false,
    });
    const message = result.messages.find(message => message.id === cursor);
    if (message) {
      return message;
    }
  }

  return null;
}

// ── Thread listing ──────────────────────────────────────────────────

export async function listThreadsForResource({
  memory,
  resourceId,
  currentThreadId,
  page = 0,
  limit = 20,
  before,
  after,
}: {
  memory: RecallMemory;
  resourceId: string;
  currentThreadId: string;
  page?: number;
  limit?: number;
  before?: string;
  after?: string;
}): Promise<{
  threads: string;
  count: number;
  page: number;
  hasMore: boolean;
}> {
  if (!resourceId) {
    throw new Error('Resource ID is required to list threads');
  }

  const MAX_LIMIT = 50;
  const normalizedLimit = Math.min(Math.max(limit, 1), MAX_LIMIT);

  const hasDateFilter = !!(before || after);
  const beforeDate = before ? new Date(before) : null;
  const afterDate = after ? new Date(after) : null;

  // When date filtering, fetch all threads and filter client-side
  // (storage layer doesn't support date range on threads)
  const result = await memory.listThreads({
    filter: { resourceId },
    page: hasDateFilter ? 0 : page,
    perPage: hasDateFilter ? false : normalizedLimit,
    orderBy: { field: 'updatedAt', direction: 'DESC' },
  });

  let threads = result.threads;

  if (beforeDate) {
    threads = threads.filter(t => t.createdAt < beforeDate);
  }
  if (afterDate) {
    threads = threads.filter(t => t.createdAt > afterDate);
  }

  // Apply client-side pagination when date-filtered
  let hasMore: boolean;
  if (hasDateFilter) {
    const offset = page * normalizedLimit;
    hasMore = offset + normalizedLimit < threads.length;
    threads = threads.slice(offset, offset + normalizedLimit);
  } else {
    hasMore = result.hasMore;
  }

  if (threads.length === 0) {
    return {
      threads: 'No threads found matching the criteria.',
      count: 0,
      page,
      hasMore: false,
    };
  }

  const lines: string[] = [];
  for (const thread of threads) {
    const isCurrent = thread.id === currentThreadId;
    const title = thread.title || '(untitled)';
    const updated = formatTimestamp(thread.updatedAt);
    const created = formatTimestamp(thread.createdAt);
    const marker = isCurrent ? ' ← current' : '';
    lines.push(`- **${title}**${marker}`);
    lines.push(`  id: ${thread.id}`);
    lines.push(`  updated: ${updated} | created: ${created}`);
  }

  return {
    threads: lines.join('\n'),
    count: threads.length,
    page,
    hasMore,
  };
}

// ── Cross-thread search ─────────────────────────────────────────────

export const SEARCH_NOT_CONFIGURED_MESSAGE =
  'Search is not configured. Enable it with `retrieval: { vector: true }` and configure a vector store and embedder on your Memory instance.';

export async function searchMessagesForResource({
  memory,
  om,
  resourceId,
  currentThreadId,
  query,
  topK = 10,
  maxTokens = DEFAULT_MAX_RESULT_TOKENS,
  before,
  after,
  threadScope,
  currentMessages = [],
}: {
  memory: RecallMemory;
  om?: OMTimelineEngine | null;
  resourceId: string;
  currentThreadId?: string;
  query: string;
  topK?: number;
  maxTokens?: number;
  before?: string;
  after?: string;
  /** When set, restrict search results to only this thread */
  threadScope?: string;
  /** Current conversation at tool execution time, not an ever-seen history. */
  currentMessages?: readonly MastraDBMessage[];
}): Promise<{
  results: string;
  count: number;
}> {
  if (!memory.searchMessages) {
    return {
      results: SEARCH_NOT_CONFIGURED_MESSAGE,
      count: 0,
    };
  }

  const MAX_TOPK = 20;
  const clampedTopK = Math.min(Math.max(topK, 1), MAX_TOPK);
  const effectiveTopK = threadScope || before || after ? Math.max(clampedTopK * 3, clampedTopK + 10) : clampedTopK;
  const searchTopK = Math.min(MAX_TOPK, effectiveTopK);

  const beforeDate = before ? new Date(before) : undefined;
  const afterDate = after ? new Date(after) : undefined;

  const { results } = await memory.searchMessages({
    query,
    resourceId,
    topK: searchTopK,
    filter: {
      ...(threadScope ? { threadId: threadScope } : {}),
      ...(afterDate ? { observedAfter: afterDate } : {}),
      ...(beforeDate ? { observedBefore: beforeDate } : {}),
    },
  });

  if (results.length === 0) {
    return {
      results: 'No matching messages found.',
      count: 0,
    };
  }

  const threadIds = [...new Set(results.map(r => r.threadId))];
  const threadMap = new Map<string, RecallThread>();
  if (memory.getThreadById) {
    await Promise.all(
      threadIds.map(async id => {
        const thread = await memory.getThreadById!({ threadId: id });
        if (thread) threadMap.set(id, thread);
      }),
    );
  }

  const filteredMatches = results.filter(match => {
    if (threadScope && match.threadId !== threadScope) return false;
    if (beforeDate && match.observedAt && match.observedAt >= beforeDate) return false;
    if (afterDate && match.observedAt && match.observedAt <= afterDate) return false;
    return true;
  });

  if (filteredMatches.length === 0) {
    return { results: 'No matching messages found.', count: 0 };
  }

  const limitedMatches = filteredMatches.slice(0, clampedTopK);
  const contentBudget = Math.max(0, Math.floor(maxTokens));
  const perGroupBudget = Math.floor(contentBudget / limitedMatches.length);
  const now = new Date();

  // Reserve an excerpt for every hit before distributing spare space in similarity order.
  // Metadata and navigation do not consume the observation-text allowance.
  const ordered = await Promise.all(
    limitedMatches.map(async (match, index) => {
      const timeline =
        om && match.groupId && threadMap.get(match.threadId)?.resourceId === resourceId
          ? await findGroupTimeline(om, match.threadId, resourceId, match.groupId)
          : null;
      const timeZone = resolveTimeZone(timeline?.record.observedTimezone ?? undefined);
      const rawBody = (match.text || '').trim() || '_Observation text unavailable._';
      const body = match.groupId ? addRelativeTimeToObservations(rawBody, now, timeZone) : rawBody;
      return { match, index, timeline, body, excerpt: chunkTextByTokens(body, perGroupBudget).text, suppression: '' };
    }),
  );
  let remaining = contentBudget - ordered.reduce((sum, entry) => sum + estimateTokenCount(entry.excerpt), 0);
  for (const entry of ordered) {
    if (remaining <= 0) break;
    if (entry.excerpt === entry.body) continue;
    const previousTokens = estimateTokenCount(entry.excerpt);
    entry.excerpt = chunkTextByTokens(entry.body, previousTokens + remaining).text;
    remaining -= estimateTokenCount(entry.excerpt) - previousTokens;
  }

  const visibleExcerpts = getVisibleSearchExcerpts(currentMessages);
  await Promise.all(
    ordered.map(async entry => {
      if (!entry.match.groupId) return;
      if (
        entry.excerpt &&
        visibleExcerpts.get(searchContextKey(entry.match))?.some(text => text.startsWith(entry.excerpt))
      ) {
        entry.suppression = 'Excerpt already in current context.';
      } else if (await sourceRangeIsVisible({ match: entry.match, messages: currentMessages, memory, resourceId })) {
        entry.suppression = 'Source messages already in current context.';
      }
      if (entry.suppression) entry.excerpt = '';
    }),
  );
  // Reuse the space from compact references without making a previously covered excerpt longer.
  remaining = contentBudget - ordered.reduce((sum, entry) => sum + estimateTokenCount(entry.excerpt), 0);
  for (const entry of ordered) {
    if (remaining <= 0) break;
    if (entry.suppression || entry.excerpt === entry.body) continue;
    const previousTokens = estimateTokenCount(entry.excerpt);
    entry.excerpt = chunkTextByTokens(entry.body, previousTokens + remaining).text;
    remaining -= estimateTokenCount(entry.excerpt) - previousTokens;
  }

  // Chronology controls presentation, not which hits get the available text budget.
  ordered.sort((a, b) => {
    const at = a.match.observedAt?.getTime() ?? Number.POSITIVE_INFINITY;
    const bt = b.match.observedAt?.getTime() ?? Number.POSITIVE_INFINITY;
    if (at !== bt) return at < bt ? -1 : 1;
    if (a.match.threadId !== b.match.threadId) return a.match.threadId.localeCompare(b.match.threadId);
    if (Boolean(a.timeline) !== Boolean(b.timeline)) return a.timeline ? -1 : 1;
    if (a.timeline && b.timeline) {
      return (
        a.timeline.record.generationCount - b.timeline.record.generationCount ||
        a.timeline.indexById.get(a.match.groupId!)! - b.timeline.indexById.get(b.match.groupId!)!
      );
    }
    return a.index - b.index;
  });
  const timelines = ordered.map(entry => entry.timeline);

  const sections: string[] = [];
  for (let i = 0; i < ordered.length; i++) {
    const { match, body, excerpt, suppression } = ordered[i]!;
    const timeline = timelines[i]!;
    const thread = threadMap.get(match.threadId);
    const title = thread?.title || '(untitled)';
    const isCurrentThread = match.threadId === currentThreadId;
    const generationLabel = isCurrentThread ? 'Current thread memory' : 'Memory from another thread';
    const generationDetail = isCurrentThread
      ? 'This result came from the current thread.'
      : 'This result came from another thread.';
    const threadLine = `- thread: ${match.threadId}${thread ? ` (${title})` : ''}`;
    const timeZone = resolveTimeZone(timeline?.record.observedTimezone ?? undefined);
    const observedLine = match.observedAt
      ? `- observed: ${formatTimestamp(match.observedAt)} (${formatRelativeTime(match.observedAt, now, timeZone)})`
      : undefined;
    const sourceLine = match.range
      ? `- source: raw messages from ID ${match.range.split(':')[0] ?? '(unknown)'} through ID ${match.range.split(':')[1] ?? '(unknown)'}`
      : '- source: raw message range unavailable';
    const groupLine = match.groupId ? `- observation group: ${match.groupId}` : undefined;
    const scoreLine = `- score: ${match.score.toFixed(2)}`;

    if (i > 0) {
      const prev = ordered[i - 1]!;
      const marker = gapMarkerBetween(
        { threadId: prev.match.threadId, groupId: prev.match.groupId },
        { threadId: match.threadId, groupId: match.groupId },
        timelines[i - 1]!,
        timeline,
        !threadScope,
      );
      if (marker) sections.push(marker);
    }

    if (suppression) {
      sections.push(`${groupLine}\n  thread: ${match.threadId}; ${suppression}`);
      continue;
    }
    sections.push(
      [
        `### ${generationLabel}`,
        '',
        generationDetail,
        threadLine,
        observedLine,
        sourceLine,
        groupLine,
        scoreLine,
        '',
        '```text',
        excerpt,
        '```',
        excerpt !== body ? '[Excerpt truncated]' : undefined,
      ]
        .filter(Boolean)
        .join('\n'),
    );
  }

  // Edge markers: how much history lies outside the result set. Only exact when every
  // resolved hit shares one generation; otherwise the model can page from an edge hit.
  const resolved = ordered
    .map((entry, i) => ({ entry, timeline: timelines[i] }))
    .filter(item => item.timeline && item.entry.match.groupId);
  const singleTimeline =
    resolved.length === ordered.length &&
    resolved.every(
      item =>
        item.timeline!.record.id === resolved[0]!.timeline!.record.id &&
        item.entry.match.threadId === resolved[0]!.entry.match.threadId,
    );
  if (singleTimeline) {
    const timeline = resolved[0]!.timeline!;
    const firstIndex = timeline.indexById.get(ordered[0]!.match.groupId ?? '');
    const lastIndex = timeline.indexById.get(ordered.at(-1)!.match.groupId ?? '');
    if (firstIndex !== undefined && firstIndex > 0 && ordered[0]!.match.groupId) {
      sections.unshift(
        `— ${firstIndex}${timeline.record.generationCount > 0 ? '+' : ''} earlier observation groups not shown — page back with ${pagingCall(ordered[0]!.match.groupId, 'before', threadScope ? undefined : ordered[0]!.match.threadId)} —`,
      );
    }
    if (lastIndex !== undefined && lastIndex < timeline.groups.length - 1 && ordered.at(-1)!.match.groupId) {
      const afterCount = timeline.groups.length - lastIndex - 1;
      sections.push(
        `— ${afterCount}+ later observation groups not shown — page forward with ${pagingCall(ordered.at(-1)!.match.groupId!, 'after', threadScope ? undefined : ordered.at(-1)!.match.threadId)} —`,
      );
    }
  }

  if (ordered.some(entry => !entry.suppression && entry.excerpt !== entry.body)) {
    sections.push(
      [
        `All ${ordered.length} selected matching groups are shown; some excerpts are truncated.`,
        om
          ? 'To read a full group, use mode="observations" with its threadId and groupId, omitting direction.'
          : undefined,
        'Use mode="messages" with a source-range cursor for original messages.',
      ]
        .filter(Boolean)
        .join(' '),
    );
  }

  return {
    results: sections.join('\n\n'),
    count: limitedMatches.length,
  };
}

// ── Per-part formatting ─────────────────────────────────────────────

const LOW_DETAIL_PART_TOKENS = 30;
const AUTO_EXPAND_TEXT_TOKENS = 100;
const AUTO_EXPAND_TOOL_TOKENS = 20;
const HIGH_DETAIL_TOOL_RESULT_TOKENS = 4000;
const DEFAULT_MAX_RESULT_TOKENS = 2000;

function formatTimestamp(date: Date): string {
  return date
    .toISOString()
    .replace('T', ' ')
    .replace(/\.\d{3}Z$/, 'Z');
}

interface FormattedPart {
  messageId: string;
  partIndex: number;
  role: string;
  type: string;
  text: string;
  /** Full untruncated text — used for auto-expand when token budget allows */
  fullText: string;
  toolName?: string;
}

function truncateByTokens(text: string, maxTokens: number, hint?: string): { text: string; wasTruncated: boolean } {
  if (estimateTokenCount(text) <= maxTokens) return { text, wasTruncated: false };
  // Truncate content to maxTokens, then append hint outside the budget
  const truncated = truncateStringByTokens(text, maxTokens);
  const suffix = hint ? ` [${hint} for more]` : '';
  return { text: truncated + suffix, wasTruncated: true };
}

function chunkTextByTokens(
  text: string,
  maxTokens: number,
  charOffset = 0,
): { text: string; nextCharOffset?: number; charOffset: number; truncated: boolean } {
  let startOffset = Math.max(0, Math.min(Math.floor(charOffset), text.length));
  // Caller-provided offsets can land between the two halves of a surrogate
  // pair; skip the lone low surrogate so the chunk stays valid JSON text.
  const startCode = text.charCodeAt(startOffset);
  if (startCode >= 0xdc00 && startCode <= 0xdfff) startOffset += 1;
  const remaining = text.slice(startOffset);

  if (!remaining || estimateTokenCount(remaining) <= maxTokens) {
    return { text: remaining, charOffset: startOffset, truncated: false };
  }

  let low = 0;
  let high = remaining.length;
  let best = '';

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const candidate = safeSlice(remaining, mid);
    const candidateTokens = estimateTokenCount(candidate);

    if (candidate && candidateTokens <= maxTokens) {
      best = candidate;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  const nextCharOffset = startOffset + best.length;
  return {
    text: best,
    charOffset: startOffset,
    nextCharOffset,
    truncated: nextCharOffset < text.length,
  };
}

function lowDetailPartLimit(type: string): number {
  if (type === 'text') return AUTO_EXPAND_TEXT_TOKENS;
  if (type === 'tool-result' || type === 'tool-call') return AUTO_EXPAND_TOOL_TOKENS;
  return LOW_DETAIL_PART_TOKENS;
}

function makePart(
  msg: MastraDBMessage,
  partIndex: number,
  type: string,
  fullText: string,
  detail: RecallDetail,
  toolName?: string,
): FormattedPart {
  if (detail === 'high') {
    return { messageId: msg.id, partIndex, role: msg.role, type, text: fullText, fullText, toolName };
  }
  const hint = `recall cursor="${msg.id}" partIndex=${partIndex} detail="high"`;
  const { text } = truncateByTokens(fullText, lowDetailPartLimit(type), hint);
  return { messageId: msg.id, partIndex, role: msg.role, type, text, fullText, toolName };
}

function formatMessageParts(msg: MastraDBMessage, detail: RecallDetail): FormattedPart[] {
  const parts: FormattedPart[] = [];

  if (typeof msg.content === 'string') {
    parts.push(makePart(msg, 0, 'text', msg.content, detail));
    return parts;
  }

  const messageParts = getMessageParts(msg);
  if (messageParts.length > 0) {
    for (let i = 0; i < messageParts.length; i++) {
      const part = messageParts[i]!;
      const partType = (part as { type?: string }).type;

      if (partType === 'text') {
        const text = (part as { text?: string }).text;
        if (text) {
          parts.push(makePart(msg, i, 'text', text, detail));
        }
      } else if (partType === 'tool-invocation') {
        const inv = (part as any).toolInvocation;
        if (inv?.toolName) {
          const hasArgs = inv.args != null;
          if (inv.state !== 'partial-call' && hasArgs) {
            const argsStr = detail === 'low' ? '' : `\n${JSON.stringify(inv.args, null, 2)}`;
            const fullText = `[Tool Call: ${inv.toolName}]${argsStr}`;
            parts.push({
              messageId: msg.id,
              partIndex: i,
              role: msg.role,
              type: 'tool-call',
              text: fullText,
              fullText,
              toolName: inv.toolName,
            });
          }

          if (inv.state === 'result') {
            const { value: resultValue } = resolveToolResultValue(
              part as { providerMetadata?: Record<string, any> },
              inv.result,
            );
            const resultStr = formatToolResultForObserver(resultValue, { maxTokens: HIGH_DETAIL_TOOL_RESULT_TOKENS });
            const fullText = `[Tool Result: ${inv.toolName}]\n${resultStr}`;
            parts.push(makePart(msg, i, 'tool-result', fullText, detail, inv.toolName));
          }
        }
      } else if (partType === 'tool-call') {
        const toolName = (part as any).toolName;
        if (toolName) {
          const rawArgs = (part as any).input ?? (part as any).args;
          const argsStr =
            detail === 'low' || rawArgs == null
              ? ''
              : `\n${typeof rawArgs === 'string' ? rawArgs : JSON.stringify(rawArgs, null, 2)}`;
          const fullText = `[Tool Call: ${toolName}]${argsStr}`;
          parts.push({
            messageId: msg.id,
            partIndex: i,
            role: msg.role,
            type: 'tool-call',
            text: fullText,
            fullText,
            toolName,
          });
        }
      } else if (partType === 'tool-result') {
        const toolName = (part as any).toolName;
        if (toolName) {
          const rawResult = (part as any).output ?? (part as any).result;
          const resultStr = formatToolResultForObserver(rawResult, { maxTokens: HIGH_DETAIL_TOOL_RESULT_TOKENS });
          const fullText = `[Tool Result: ${toolName}]\n${resultStr}`;
          parts.push(makePart(msg, i, 'tool-result', fullText, detail, toolName));
        }
      } else if (partType === 'reasoning') {
        const reasoning = (part as { reasoning?: string; text?: string }).reasoning ?? (part as { text?: string }).text;
        if (reasoning) {
          parts.push(makePart(msg, i, 'reasoning', reasoning, detail));
        }
      } else if (partType === 'image' || partType === 'file') {
        const filename = (part as any).filename;
        const label = filename ? `: ${filename}` : '';
        const fullText = `[${partType === 'image' ? 'Image' : 'File'}${label}]`;
        parts.push({ messageId: msg.id, partIndex: i, role: msg.role, type: partType, text: fullText, fullText });
      } else if (partType?.startsWith('data-')) {
        // skip data parts — these are internal OM markers (buffering, observation, etc.)
      } else if (partType) {
        const fullText = `[${partType}]`;
        parts.push({ messageId: msg.id, partIndex: i, role: msg.role, type: partType, text: fullText, fullText });
      }
    }
  } else if (msg.content?.content) {
    parts.push(makePart(msg, 0, 'text', msg.content.content, detail));
  }

  return parts;
}

function buildRenderedText(parts: FormattedPart[], timestamps: Map<string, Date>): string {
  let currentMessageId = '';
  const lines: string[] = [];

  for (const part of parts) {
    if (part.messageId !== currentMessageId) {
      currentMessageId = part.messageId;
      const ts = timestamps.get(part.messageId);
      const tsStr = ts ? ` (${formatTimestamp(ts)})` : '';
      if (lines.length > 0) lines.push(''); // blank line between messages
      lines.push(`**${part.role}${tsStr}** [${part.messageId}]:`);
    }

    const indexLabel = `[p${part.partIndex}]`;
    lines.push(`  ${indexLabel} ${part.text}`);
  }

  return lines.join('\n');
}

async function getNextVisibleMessage({
  memory,
  threadId,
  resourceId,
  after,
}: {
  memory: RecallMemory;
  threadId: string;
  resourceId?: string;
  after: Date;
}): Promise<MastraDBMessage | null> {
  const result = await memory.recall({
    threadId,
    resourceId,
    page: 0,
    perPage: 50,
    orderBy: { field: 'createdAt', direction: 'ASC' },
    filter: {
      dateRange: {
        start: after,
        startExclusive: true,
      },
    },
  });

  return result.messages.find(hasVisibleParts) ?? null;
}

const MAX_EXPAND_USER_TEXT_TOKENS = 200;
const MAX_EXPAND_OTHER_TOKENS = 50;

function expandLimit(part: FormattedPart): number {
  if (part.role === 'user' && part.type === 'text') return MAX_EXPAND_USER_TEXT_TOKENS;
  return MAX_EXPAND_OTHER_TOKENS;
}

function expandPriority(part: FormattedPart): number {
  // Lower number = higher priority for expansion
  if (part.role === 'user' && part.type === 'text') return 0;
  if (part.type === 'text' || part.type === 'reasoning') return 1;
  if (part.type === 'tool-result') return 2;
  if (part.type === 'tool-call') return 3;
  return 4;
}

function renderFormattedParts(
  parts: FormattedPart[],
  timestamps: Map<string, Date>,
  options: { detail: RecallDetail; maxTokens: number },
): { text: string; truncated: boolean; tokenOffset: number } {
  // Step 1: render with per-part truncated text
  const text = buildRenderedText(parts, timestamps);
  let totalTokens = estimateTokenCount(text);

  if (totalTokens > options.maxTokens) {
    // Already over budget even with truncated text — hard-truncate
    const truncated = truncateStringByTokens(text, options.maxTokens);
    return { text: truncated, truncated: true, tokenOffset: totalTokens - options.maxTokens };
  }

  // Step 2: we're under budget — try expanding truncated parts with leftover room.
  // Find parts where text !== fullText (i.e., they were truncated).
  const truncatedIndices = parts
    .map((p, i) => ({ part: p, index: i }))
    .filter(({ part }) => part.text !== part.fullText)
    .sort((a, b) => expandPriority(a.part) - expandPriority(b.part));

  if (truncatedIndices.length === 0) {
    return { text, truncated: false, tokenOffset: 0 };
  }

  let remaining = options.maxTokens - totalTokens;

  for (const { part, index } of truncatedIndices) {
    if (remaining <= 0) break;

    const maxTokens = expandLimit(part);
    const fullTokens = estimateTokenCount(part.fullText);
    const currentTokens = estimateTokenCount(part.text);
    // Cap at the expand limit for this part type
    const targetTokens = Math.min(fullTokens, maxTokens);
    const delta = targetTokens - currentTokens;

    if (delta <= 0) continue; // already at or above expand limit

    if (delta <= remaining && targetTokens >= fullTokens) {
      // Full text fits within both expand limit and remaining budget
      parts[index] = { ...part, text: part.fullText };
      remaining -= delta;
    } else {
      // Partial expand — cap at expand limit or remaining budget, whichever is smaller
      const expandedLimit = Math.min(currentTokens + remaining, maxTokens);
      const hint = `recall cursor="${part.messageId}" partIndex=${part.partIndex} detail="high"`;
      const { text: expanded } = truncateByTokens(part.fullText, expandedLimit, hint);
      const expandedDelta = estimateTokenCount(expanded) - currentTokens;
      parts[index] = { ...part, text: expanded };
      remaining -= expandedDelta;
    }
  }

  // Step 3: re-render with expanded parts
  const expanded = buildRenderedText(parts, timestamps);
  const expandedTokens = estimateTokenCount(expanded);

  if (expandedTokens <= options.maxTokens) {
    return { text: expanded, truncated: false, tokenOffset: 0 };
  }

  // Safety net: if token estimates drifted, hard-truncate
  const hardTruncated = truncateStringByTokens(expanded, options.maxTokens);
  return { text: hardTruncated, truncated: true, tokenOffset: expandedTokens - options.maxTokens };
}

// ── Single-part fetch ────────────────────────────────────────────────

export async function recallPart({
  memory,
  threadId,
  resourceId,
  cursor,
  partIndex,
  charOffset,
  threadScope,
  retrievalScope = 'thread',
  maxTokens = DEFAULT_MAX_RESULT_TOKENS,
}: {
  memory: RecallMemory;
  threadId: string;
  resourceId?: string;
  cursor: string;
  partIndex: number;
  charOffset?: number;
  threadScope?: string;
  retrievalScope?: 'thread' | 'resource';
  maxTokens?: number;
}): Promise<{
  text: string;
  messageId: string;
  partIndex: number;
  role: string;
  type: string;
  truncated: boolean;
  charOffset: number;
  nextCharOffset?: number;
  note?: string;
}> {
  if (!memory || typeof memory.getMemoryStore !== 'function') {
    throw new Error('Memory instance is required for recall');
  }

  if (!threadId) {
    throw new Error('Thread ID is required for recall');
  }

  const resolved = await resolveCursorMessage(memory, cursor, {
    resourceId,
    threadScope,
    enforceThreadScope: retrievalScope !== 'resource',
  });

  if ('hint' in resolved) {
    throw new Error(resolved.hint);
  }

  const allParts = formatMessageParts(resolved, 'high');

  if (allParts.length === 0) {
    throw new Error(
      `Message ${cursor} has no visible content (it may be an internal system message). Try a neighboring message ID instead.`,
    );
  }

  const target = [...allParts].reverse().find(p => p.partIndex === partIndex);

  if (!target) {
    const availableIndices = allParts.map(p => p.partIndex).join(', ');
    const highestVisiblePartIndex = Math.max(...allParts.map(p => p.partIndex));

    if (partIndex > highestVisiblePartIndex) {
      const nextMessage = await getNextVisibleMessage({
        memory,
        threadId: resolved.threadId ?? threadId,
        resourceId,
        after: resolved.createdAt,
      });

      if (nextMessage) {
        const nextParts = formatMessageParts(nextMessage, 'high');
        const firstNextPart = nextParts[0];

        if (firstNextPart) {
          const fallbackNote = `Part index ${partIndex} not found in message ${cursor}; showing partIndex ${firstNextPart.partIndex} from next message ${firstNextPart.messageId}.\n\n`;
          const fallbackText = `${fallbackNote}${firstNextPart.text}`;
          const fallbackChunk = chunkTextByTokens(fallbackText, maxTokens, charOffset);
          const fallbackContinuation = fallbackChunk.nextCharOffset
            ? `To continue this part, call recall cursor="${cursor}" partIndex=${partIndex} detail="high" charOffset=${fallbackChunk.nextCharOffset}.`
            : undefined;

          return {
            text: fallbackChunk.text,
            messageId: firstNextPart.messageId,
            partIndex: firstNextPart.partIndex,
            role: firstNextPart.role,
            type: firstNextPart.type,
            truncated: fallbackChunk.truncated,
            charOffset: fallbackChunk.charOffset,
            nextCharOffset: fallbackChunk.nextCharOffset,
            note: fallbackContinuation,
          };
        }
      }
    }

    throw new Error(`Part index ${partIndex} not found in message ${cursor}. Available indices: ${availableIndices}`);
  }

  const chunk = chunkTextByTokens(target.text, maxTokens, charOffset);
  const note = chunk.nextCharOffset
    ? `To continue this part, call recall cursor="${target.messageId}" partIndex=${target.partIndex} detail="high" charOffset=${chunk.nextCharOffset}.`
    : undefined;

  return {
    text: chunk.text,
    messageId: target.messageId,
    partIndex: target.partIndex,
    role: target.role,
    type: target.type,
    truncated: chunk.truncated,
    charOffset: chunk.charOffset,
    nextCharOffset: chunk.nextCharOffset,
    note,
  };
}

// ── Paged recall ─────────────────────────────────────────────────────

export interface RecallResult {
  messages: string;
  count: number;
  cursor: string;
  page: number;
  limit: number;
  detail: RecallDetail;
  hasNextPage: boolean;
  hasPrevPage: boolean;
  truncated: boolean;
  tokenOffset: number;
}

export async function recallMessages({
  memory,
  threadId,
  resourceId,
  cursor,
  page = 1,
  limit = 20,
  detail = 'low',
  partType,
  toolName,
  threadScope,
  retrievalScope = 'thread',
  maxTokens = DEFAULT_MAX_RESULT_TOKENS,
}: {
  memory: RecallMemory;
  threadId: string;
  resourceId?: string;
  cursor: string;
  page?: number;
  limit?: number;
  detail?: RecallDetail;
  partType?: 'text' | 'tool-call' | 'tool-result' | 'reasoning' | 'image' | 'file';
  toolName?: string;
  threadScope?: string;
  retrievalScope?: 'thread' | 'resource';
  maxTokens?: number;
}): Promise<RecallResult> {
  if (!memory) {
    throw new Error('Memory instance is required for recall');
  }

  if (!threadId) {
    throw new Error('Thread ID is required for recall');
  }

  if (typeof memory.getMemoryStore !== 'function') {
    throw new Error('recall requires a Memory instance with storage access');
  }

  const MAX_PAGE = 50;
  const MAX_LIMIT = 20;
  const rawPage = page === 0 ? 1 : page;
  const normalizedPage = Math.max(Math.min(rawPage, MAX_PAGE), -MAX_PAGE);
  const normalizedLimit = Math.min(limit, MAX_LIMIT);

  const resolved = await resolveCursorMessage(memory, cursor, {
    resourceId,
    threadScope,
    enforceThreadScope: retrievalScope === 'thread',
  });

  if ('hint' in resolved) {
    return {
      messages: resolved.hint,
      count: 0,
      cursor,
      page: normalizedPage,
      limit: normalizedLimit,
      detail,
      hasNextPage: false,
      hasPrevPage: false,
      truncated: false,
      tokenOffset: 0,
    };
  }

  const anchor = resolved;
  const crossThreadId = anchor.threadId && anchor.threadId !== threadId ? anchor.threadId : undefined;

  if (crossThreadId && threadScope) {
    return {
      messages: `Cursor does not belong to the active thread. Expected thread "${threadId}" but cursor "${cursor}" belongs to "${anchor.threadId}". Pass threadId="${anchor.threadId}" to browse that thread.`,
      count: 0,
      cursor,
      page: normalizedPage,
      limit: normalizedLimit,
      detail,
      hasNextPage: false,
      hasPrevPage: false,
      truncated: false,
      tokenOffset: 0,
    };
  }

  const resolvedThreadId = crossThreadId ?? threadId;
  if (!resolvedThreadId) {
    throw new Error('Thread ID is required for recall');
  }

  const isForward = normalizedPage > 0;
  const pageIndex = Math.max(Math.abs(normalizedPage), 1) - 1;
  const skip = pageIndex * normalizedLimit;

  // Fetch skip + limit + 1 to detect whether another page exists beyond this one
  const fetchCount = skip + normalizedLimit + 1;

  const result = await memory.recall({
    threadId: resolvedThreadId,
    resourceId,
    page: 0,
    perPage: fetchCount,
    orderBy: { field: 'createdAt', direction: isForward ? 'ASC' : 'DESC' },
    filter: {
      dateRange: isForward
        ? {
            start: anchor.createdAt,
            startExclusive: true,
          }
        : {
            end: anchor.createdAt,
            endExclusive: true,
          },
    },
  });

  // Filter out messages with only internal data-* parts so they don't consume page slots.
  const visibleMessages = result.messages.filter(hasVisibleParts);

  // Memory.recall() always returns messages sorted chronologically (ASC) via MessageList.
  // For forward pagination: take from the start of the ASC array (oldest first after cursor).
  // For backward pagination: take from the END of the ASC array (closest to cursor).
  //   DESC query ensures the DB returns the N messages closest to cursor, but MessageList
  //   re-sorts them to ASC. So we slice from the end to get the right page window.
  const total = visibleMessages.length;
  const hasMore = total > skip + normalizedLimit;
  let messages: typeof visibleMessages;
  if (isForward) {
    messages = visibleMessages.slice(skip, skip + normalizedLimit);
  } else {
    // For backward: closest-to-cursor messages are at the end of the ASC-sorted array.
    // Page -1 (skip=0): last `limit` items; page -2 (skip=limit): next `limit` from end; etc.
    const endIdx = Math.max(total - skip, 0);
    const startIdx = Math.max(endIdx - normalizedLimit, 0);
    messages = visibleMessages.slice(startIdx, endIdx);
  }

  // Compute pagination flags
  const hasNextPage = isForward ? hasMore : pageIndex > 0;
  const hasPrevPage = isForward ? pageIndex > 0 : hasMore;

  // Format parts from returned messages
  let allParts: FormattedPart[] = [];
  const timestamps = new Map<string, Date>();
  for (const msg of messages) {
    timestamps.set(msg.id, msg.createdAt);
    allParts.push(...formatMessageParts(msg, detail));
  }

  if (toolName) {
    allParts = allParts.filter(p => (p.type === 'tool-call' || p.type === 'tool-result') && p.toolName === toolName);
  }

  if (partType) {
    allParts = allParts.filter(p => p.type === partType);
  }

  // High detail: clamp to 1 message and 1 part to avoid token blowup
  if (detail === 'high' && allParts.length > 0) {
    const firstPart = allParts[0]!;
    const sameMsgParts = allParts.filter(p => p.messageId === firstPart.messageId);
    const otherMsgParts = allParts.filter(p => p.messageId !== firstPart.messageId);

    const rendered = renderFormattedParts([firstPart], timestamps, { detail, maxTokens });

    let text = rendered.text;

    // Build continuation hints
    const hints: string[] = [];
    if (sameMsgParts.length > 1) {
      const nextPart = sameMsgParts[1]!;
      hints.push(`next part: partIndex=${nextPart.partIndex} on cursor="${firstPart.messageId}"`);
    }
    if (otherMsgParts.length > 0) {
      const next = otherMsgParts[0]!;
      hints.push(`next message: partIndex=${next.partIndex} on cursor="${next.messageId}"`);
    } else if (hasNextPage) {
      hints.push(`more messages available on page ${normalizedPage + 1}`);
    }

    if (hints.length > 0) {
      text += `\n\nHigh detail returns 1 part at a time. To continue: ${hints.join(', or ')}.`;
    }

    return {
      messages: text,
      count: 1,
      cursor,
      page: normalizedPage,
      limit: normalizedLimit,
      detail,
      hasNextPage: otherMsgParts.length > 0 || hasNextPage,
      hasPrevPage,
      truncated: rendered.truncated,
      tokenOffset: rendered.tokenOffset,
    };
  }

  const rendered = renderFormattedParts(allParts, timestamps, { detail, maxTokens });
  const emptyMessage =
    allParts.length === 0
      ? partType || toolName
        ? '(no message parts matched the current filters)'
        : '(no visible message parts found for this page)'
      : '(no messages found)';

  return {
    messages: rendered.text || emptyMessage,
    count: messages.length,
    cursor,
    page: normalizedPage,
    limit: normalizedLimit,
    detail,
    hasNextPage,
    hasPrevPage,
    truncated: rendered.truncated,
    tokenOffset: rendered.tokenOffset,
  };
}

// ── Thread browsing (no cursor) ─────────────────────────────────────

export async function recallThreadFromStart({
  memory,
  threadId,
  resourceId,
  page = 1,
  limit = 20,
  detail = 'low',
  partType,
  toolName,
  anchor = 'start',
  maxTokens = DEFAULT_MAX_RESULT_TOKENS,
}: {
  memory: RecallMemory;
  threadId: string;
  resourceId?: string;
  page?: number;
  limit?: number;
  detail?: RecallDetail;
  partType?: 'text' | 'tool-call' | 'tool-result' | 'reasoning' | 'image' | 'file';
  toolName?: string;
  anchor?: 'start' | 'end';
  maxTokens?: number;
}): Promise<RecallResult> {
  if (!memory) {
    throw new Error('Memory instance is required for recall');
  }
  if (!threadId) {
    throw new Error('Thread ID is required for recall');
  }

  // Verify the thread belongs to the current resource
  if (resourceId && memory.getThreadById) {
    const thread = await memory.getThreadById({ threadId });
    if (!thread || thread.resourceId !== resourceId) {
      throw new Error('Thread not found');
    }
  }

  const MAX_PAGE = 50;
  const MAX_LIMIT = 20;
  const normalizedPage = Math.max(Math.min(page, MAX_PAGE), 1);
  const normalizedLimit = Math.min(Math.max(limit, 1), MAX_LIMIT);
  const pageIndex = normalizedPage - 1;
  const fetchCount = pageIndex * normalizedLimit + normalizedLimit + 1;

  const result = await memory.recall({
    threadId,
    resourceId,
    page: 0,
    perPage: fetchCount,
    orderBy: { field: 'createdAt', direction: anchor === 'end' ? 'DESC' : 'ASC' },
  });

  const visibleMessages =
    anchor === 'end'
      ? result.messages.slice(0, fetchCount).filter(hasVisibleParts).reverse()
      : result.messages.slice(0, fetchCount).filter(hasVisibleParts);
  const skip = pageIndex * normalizedLimit;
  const messages = visibleMessages.slice(skip, skip + normalizedLimit);
  const hasExtraMessage = visibleMessages.length > skip + messages.length;
  const hasNextPage = messages.length > 0 ? (anchor === 'end' ? pageIndex > 0 : hasExtraMessage) : false;
  const hasPrevPage = messages.length > 0 ? (anchor === 'end' ? hasExtraMessage : pageIndex > 0) : pageIndex > 0;

  let allParts: FormattedPart[] = [];
  const timestamps = new Map<string, Date>();
  for (const msg of messages) {
    timestamps.set(msg.id, msg.createdAt);
    allParts.push(...formatMessageParts(msg, detail));
  }

  if (toolName) {
    allParts = allParts.filter(p => (p.type === 'tool-call' || p.type === 'tool-result') && p.toolName === toolName);
  }

  if (partType) {
    allParts = allParts.filter(p => p.type === partType);
  }

  const rendered = renderFormattedParts(allParts, timestamps, { detail, maxTokens });
  const emptyMessage =
    messages.length === 0
      ? pageIndex > 0
        ? `(no messages found on page ${normalizedPage} for this thread)`
        : '(no messages in this thread)'
      : partType || toolName
        ? '(no message parts matched the current filters)'
        : '(no messages found)';

  return {
    messages: rendered.text || emptyMessage,
    count: messages.length,
    cursor: messages[0]?.id || '',
    page: normalizedPage,
    limit: normalizedLimit,
    detail,
    hasNextPage,
    hasPrevPage,
    truncated: rendered.truncated,
    tokenOffset: rendered.tokenOffset,
  };
}

export const recallTool = (
  _memoryConfig?: MemoryConfigInternal,
  options?: {
    retrievalScope?: 'thread' | 'resource';
    searchEnabled?: boolean;
    /** Resolves the observational memory engine so search hits can be dated and paged by observation group. */
    getOMEngine?: () => Promise<OMTimelineEngine | null> | OMTimelineEngine | null;
  },
) => {
  const retrievalScope = options?.retrievalScope ?? 'thread';
  const isResourceScope = retrievalScope === 'resource';
  const searchEnabled = options?.searchEnabled ?? true;
  const getOMEngine = options?.getOMEngine;

  const description = isResourceScope
    ? `Browse conversation history. Use mode="threads" to list all threads for the current user. Use mode="messages" (default) to browse messages in the current thread or pass threadId to browse another thread in the active resource. When mode="messages" has no cursor or threadId, it defaults to the current thread and says so at the top of the result. If you pass only a cursor, it must belong to the current thread.${searchEnabled ? ' Use mode="search" to find messages by content across all threads; results come back oldest-first with markers where observation groups were skipped. Use mode="observations" with a groupId from a search result to page chronologically through the observation groups around it.' : ''}`
    : `Browse conversation history in the current thread. Use mode="messages" (default) to page through messages near a cursor.${searchEnabled ? ' Use mode="search" to find messages by content in this thread; results come back oldest-first with markers where observation groups were skipped. Use mode="observations" with a groupId from a search result to page chronologically through the observation groups around it.' : ''} Use mode="threads" to get the current thread's ID and title.`;

  const modeEnum = searchEnabled ? ['messages', 'threads', 'search', 'observations'] : ['messages', 'threads'];

  return createTool({
    id: 'recall',
    description,
    inputSchema: {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'object',
      properties: {
        ...(isResourceScope
          ? {
              mode: {
                type: 'string',
                enum: modeEnum,
                description: `What to retrieve. "messages" (default) pages through message history. "threads" lists all threads for the current user.${searchEnabled ? ' "search" finds messages by semantic similarity across all threads. "observations" pages observation groups chronologically around a groupId from a search result.' : ''}`,
              },
              threadId: {
                type: 'string',
                minLength: 1,
                description:
                  'Browse a different thread, or "current" for the active thread. Use mode="threads" first to discover thread IDs.',
              },
              before: {
                type: 'string',
                description:
                  'Only show results before this date. Applies to mode="threads" (thread creation date) and mode="search" (observation date). ISO 8601 or natural date string (e.g. "2026-03-15", "2026-03-10T00:00:00Z").',
              },
              after: {
                type: 'string',
                description:
                  'Only show results after this date. Applies to mode="threads" (thread creation date) and mode="search" (observation date). ISO 8601 or natural date string (e.g. "2026-03-01", "2026-03-10T00:00:00Z").',
              },
            }
          : {
              mode: {
                type: 'string',
                enum: modeEnum,
                description: `What to retrieve. "messages" (default) pages through message history. "threads" returns info about the current thread.${searchEnabled ? ' "search" finds messages by semantic similarity in this thread. "observations" pages observation groups chronologically around a groupId from a search result.' : ''}`,
              },
            }),
        ...(searchEnabled
          ? {
              ...(!isResourceScope
                ? {
                    before: {
                      type: 'string',
                      description: 'For mode="search": only show observations dated before this ISO 8601 date.',
                    },
                    after: {
                      type: 'string',
                      description: 'For mode="search": only show observations dated after this ISO 8601 date.',
                    },
                  }
                : {}),
              query: {
                type: 'string',
                minLength: 1,
                description: 'Search query for mode="search". Finds messages semantically similar to this text.',
              },
              groupId: {
                type: 'string',
                minLength: 1,
                description:
                  'Observation group ID to page around, for mode="observations". Copy it from the "observation group:" line of a search result.',
              },
              direction: {
                type: 'string',
                enum: ['before', 'after'],
                description:
                  'For mode="observations": page strictly before or after the groupId. Omit to read the full anchor group and following groups.',
              },
            }
          : {}),
        cursor: {
          type: 'string',
          minLength: 1,
          description:
            'A message ID to use as the pagination cursor. For mode="messages", omit both cursor and threadId to browse the current thread. If only cursor is provided, it must belong to the current thread. Extract it from the start or end of an observation group range.',
        },
        anchor: {
          type: 'string',
          enum: ['start', 'end'],
          description:
            'For mode="messages" without a cursor, page from the start (oldest-first) or end (newest-first) of the thread. Defaults to "start".',
        },
        page: {
          type: 'integer',
          minimum: -50,
          maximum: 50,
          description:
            'Pagination offset. For messages: positive pages move forward from cursor, negative move backward. For threads: page number (0-indexed). 0 is treated as 1 for messages.',
        },
        limit: {
          type: 'integer',
          minimum: 1,
          maximum: 20,
          description:
            'Maximum number of items to return per page. Defaults to 20 for messages, 5 for mode="observations".',
        },
        detail: {
          type: 'string',
          enum: ['low', 'high'],
          description:
            'Detail level for messages. "low" (default) returns truncated text and tool names. "high" returns full content with tool args/results.',
        },
        partType: {
          type: 'string',
          enum: ['text', 'tool-call', 'tool-result', 'reasoning', 'image', 'file'],
          description: 'Filter results to only include parts of this type. Only applies to mode="messages".',
        },
        toolName: {
          type: 'string',
          minLength: 1,
          description:
            'Filter results to only include tool-call and tool-result parts matching this tool name. Only applies to mode="messages".',
        },
        partIndex: {
          type: 'integer',
          minimum: 0,
          description:
            'Fetch a single part from the cursor message by its positional index. When provided, returns only that part at high detail. Indices are shown as [p0], [p1], etc. in recall results.',
        },
        charOffset: {
          type: 'integer',
          minimum: 0,
          description:
            'Continue reading a truncated single part from this position. Pass the exact nextCharOffset value returned by a previous call; do not compute it yourself. Only applies with cursor and partIndex in mode="messages".',
        },
      },
    } satisfies JSONSchema7,
    execute: async (inputData, context) => {
      const {
        mode,
        query,
        groupId,
        direction,
        cursor,
        threadId: explicitThreadId,
        anchor,
        page,
        limit,
        detail,
        partType,
        toolName,
        partIndex,
        charOffset,
        before,
        after,
      } = inputData as {
        mode?: 'messages' | 'threads' | 'search' | 'observations';
        query?: string;
        groupId?: string;
        direction?: 'before' | 'after';
        cursor?: string;
        threadId?: string;
        anchor?: 'start' | 'end';
        page?: number;
        limit?: number;
        detail?: RecallDetail;
        partType?: 'text' | 'tool-call' | 'tool-result' | 'reasoning' | 'image' | 'file';
        toolName?: string;
        partIndex?: number;
        charOffset?: number;
        before?: string;
        after?: string;
      };
      const memory = (context as any)?.memory as RecallMemory | undefined;
      const currentThreadId = context?.agent?.threadId;
      const resourceId = context?.agent?.resourceId;
      const resolvedExplicitThreadId = explicitThreadId === 'current' ? currentThreadId : explicitThreadId;

      if (!memory) {
        throw new Error('Memory instance is required for recall');
      }

      if (explicitThreadId === 'current' && !currentThreadId) {
        throw new Error('Could not resolve current thread.');
      }

      // Observation group paging mode
      if (mode === 'observations') {
        if (!searchEnabled) {
          return { results: SEARCH_NOT_CONFIGURED_MESSAGE, count: 0 };
        }
        if (!groupId) {
          throw new Error('groupId is required for mode="observations"');
        }
        if (!resourceId) {
          throw new Error('Resource ID is required for recall');
        }
        const om = (await getOMEngine?.()) ?? null;
        if (!om) {
          return {
            results:
              'Observation group paging requires observational memory and a storage adapter supporting observation history search.',
            count: 0,
          };
        }
        const pagingThreadId = !isResourceScope ? currentThreadId : resolvedExplicitThreadId || currentThreadId;
        if (!pagingThreadId) {
          throw new Error('Could not resolve current thread.');
        }
        const thread = await memory.getThreadById?.({ threadId: pagingThreadId });
        if (!thread || thread.resourceId !== resourceId) {
          throw new Error('Thread not found');
        }
        return pageObservationGroups({
          om,
          threadId: pagingThreadId,
          resourceId,
          groupId,
          direction,
          limit: Math.min(Math.max(limit ?? 5, 1), 20),
          threadTitle: thread.title,
          includeThreadId: isResourceScope,
        });
      }

      // Search mode
      if (mode === 'search') {
        // Schema validation rejects mode="search" when search is disabled, but
        // validation is skipped for resumed runs and builder-validated input —
        // a stale search call on those paths would otherwise reach
        // Memory.searchMessages and throw. Return guidance instead.
        if (!searchEnabled) {
          return { results: SEARCH_NOT_CONFIGURED_MESSAGE, count: 0 };
        }
        if (!query) {
          throw new Error('query is required for mode="search"');
        }
        if (!resourceId) {
          throw new Error('Resource ID is required for recall');
        }
        return searchMessagesForResource({
          memory,
          om: (await getOMEngine?.()) ?? null,
          resourceId,
          currentThreadId: currentThreadId || undefined,
          currentMessages: context?.agent?.getMessages?.(),
          query,
          topK: limit ?? 10,
          before,
          after,
          threadScope: !isResourceScope ? currentThreadId || undefined : resolvedExplicitThreadId || undefined,
        });
      }

      // Thread listing mode
      if (mode === 'threads') {
        const requestedCurrentThread = explicitThreadId === 'current';

        // Thread scope: return current thread info only
        if (!isResourceScope || requestedCurrentThread) {
          if (!currentThreadId || !memory.getThreadById) {
            return { error: 'Could not resolve current thread.' };
          }
          const thread = await memory.getThreadById({ threadId: currentThreadId });
          if (!thread) {
            return { error: 'Could not resolve current thread.' };
          }
          if (isResourceScope && resourceId && thread.resourceId !== resourceId) {
            throw new Error('Thread does not belong to the active resource');
          }
          return {
            threads: `- **${thread.title || '(untitled)'}** ← current\n  id: ${thread.id}\n  updated: ${formatTimestamp(thread.updatedAt)} | created: ${formatTimestamp(thread.createdAt)}`,
            count: 1,
            page: 0,
            hasMore: false,
          };
        }
        if (!resourceId) {
          throw new Error('Resource ID is required for recall');
        }
        return listThreadsForResource({
          memory,
          resourceId,
          currentThreadId: currentThreadId || '',
          page: page ?? 0,
          limit: limit ?? 20,
          before,
          after,
        });
      }

      const usedDefaultThreadId = !explicitThreadId && !cursor && Boolean(currentThreadId);
      const defaultThreadNote =
        usedDefaultThreadId && isResourceScope ? `threadId wasn't passed so used default ${currentThreadId}.\n\n` : '';
      // Reuse the shared `resolvedExplicitThreadId` ('current' -> currentThreadId) mapping,
      // falling back to the current thread when no threadId or cursor was provided.
      const resolvedThreadId = resolvedExplicitThreadId || (usedDefaultThreadId ? currentThreadId : undefined);
      const hasResolvedThreadId = typeof resolvedThreadId === 'string' && resolvedThreadId.length > 0;
      const hasCursor = typeof cursor === 'string' && cursor.length > 0;

      if (!hasResolvedThreadId && !hasCursor) {
        throw new Error(
          isResourceScope
            ? 'No active thread context and no cursor or threadId was provided for mode="messages". Pass a threadId (use mode="threads" to discover thread IDs) or a message ID as cursor.'
            : 'No active thread context for mode="messages". This tool is limited to the current thread and no current thread could be resolved.',
        );
      }

      let targetThreadId: string | undefined;
      let threadScope: string | undefined;

      if (!isResourceScope) {
        targetThreadId = currentThreadId;
        threadScope = currentThreadId || undefined;
      } else if (hasResolvedThreadId) {
        if (!resourceId) {
          throw new Error('Resource ID is required for recall');
        }
        if (!memory.getThreadById) {
          throw new Error('Memory instance cannot verify thread access for recall');
        }

        const thread = await memory.getThreadById({ threadId: resolvedThreadId! });
        if (!thread || thread.resourceId !== resourceId) {
          throw new Error('Thread does not belong to the active resource');
        }

        targetThreadId = thread.id;
        threadScope = thread.id;
      } else {
        targetThreadId = currentThreadId;
        threadScope = currentThreadId || undefined;
      }

      if (hasCursor && !hasResolvedThreadId && !currentThreadId) {
        if (!isResourceScope) {
          throw new Error('Current thread is required when browsing by cursor');
        }

        const resolved = await resolveCursorMessage(memory, cursor!, { resourceId });
        if ('hint' in resolved) {
          return {
            messages: resolved.hint,
            count: 0,
            cursor: cursor!,
            page: page ?? 1,
            limit: Math.min(limit ?? 20, 20),
            detail: detail ?? 'low',
            hasNextPage: false,
            hasPrevPage: false,
            truncated: false,
            tokenOffset: 0,
          };
        }

        targetThreadId = resolved.threadId;
      }

      if (!targetThreadId) {
        throw new Error('Thread ID is required for recall');
      }

      // No cursor — read from the start of the thread
      if (!cursor) {
        const result = await recallThreadFromStart({
          memory,
          threadId: targetThreadId,
          resourceId: isResourceScope ? resourceId : undefined,
          page: page ?? 1,
          limit: limit ?? 20,
          detail: detail ?? 'low',
          partType,
          toolName,
          anchor: anchor ?? 'start',
        });

        if (defaultThreadNote) {
          return { ...result, messages: `${defaultThreadNote}${result.messages}` };
        }

        return result;
      }

      // Single-part fetch mode
      if (partIndex !== undefined && partIndex !== null) {
        return recallPart({
          memory,
          threadId: targetThreadId,
          resourceId,
          cursor,
          partIndex,
          charOffset,
          threadScope,
          retrievalScope,
        });
      }

      return recallMessages({
        memory,
        threadId: targetThreadId,
        resourceId,
        cursor,
        page,
        limit,
        detail: detail ?? 'low',
        partType,
        toolName,
        threadScope,
        retrievalScope,
      });
    },
  });
};
