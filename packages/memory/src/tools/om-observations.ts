import type { ObservationalMemoryHistoryOptions, ObservationalMemoryRecord } from '@mastra/core/storage';
import xxhash from 'xxhash-wasm';
import { addRelativeTimeToObservations } from '../processors/observational-memory/date-utils';
import { parseObservationGroups } from '../processors/observational-memory/observation-groups';
import type { ObservationGroup } from '../processors/observational-memory/observation-groups';

export type OMGenerationRecord = Pick<
  ObservationalMemoryRecord,
  'id' | 'generationCount' | 'activeObservations' | 'bufferedObservationChunks' | 'observedTimezone' | 'threadId'
>;
export interface OMTimelineEngine {
  getHistory(
    threadId: string,
    resourceId: string,
    limit?: number,
    options?: ObservationalMemoryHistoryOptions,
  ): Promise<OMGenerationRecord[]>;
}
export interface GroupTimeline {
  record: OMGenerationRecord;
  groups: ObservationGroup[];
  indexById: Map<string, number>;
}
const hasher = xxhash();

async function buildTimeline(record: OMGenerationRecord, threadId: string): Promise<GroupTimeline> {
  // Buffering indexes originals before activation. Preserve the same append order as activation.
  let text = [
    record.activeObservations,
    ...(record.bufferedObservationChunks ?? []).map(chunk => chunk.observations),
  ].join('\n');
  if (record.threadId === null) {
    // Resource-scoped records mix threads; attribution can use raw or obscured thread IDs.
    const obscuredId = (await hasher).h32ToString(threadId);
    text = [...text.matchAll(/<thread id="([^"]+)">([\s\S]*?)<\/thread>/g)]
      .filter(match => match[1] === threadId || match[1] === obscuredId)
      .map(match => match[2])
      .join('\n');
  }
  const groups = parseObservationGroups(text).filter(group => group.kind !== 'reflection');
  const indexById = new Map<string, number>();
  const unique = groups.filter(group => {
    if (indexById.has(group.id)) return false;
    indexById.set(group.id, indexById.size);
    return true;
  });
  return { record, groups: unique, indexById };
}

export async function findGroupTimeline(
  om: OMTimelineEngine,
  threadId: string,
  resourceId: string,
  groupId: string,
): Promise<GroupTimeline | null> {
  const [record] = await om.getHistory(threadId, resourceId, 1, { groupId, sortDirection: 'ASC' });
  if (!record || (record.threadId !== null && record.threadId !== threadId)) return null;
  const timeline = await buildTimeline(record, threadId);
  return timeline.indexById.has(groupId) ? timeline : null;
}

export function pagingCall(groupId: string, direction: 'before' | 'after', threadId?: string): string {
  return `recall(${JSON.stringify({ mode: 'observations', threadId, groupId, direction })})`;
}

export function gapMarkerBetween(
  prev: { threadId: string; groupId?: string },
  next: { threadId: string; groupId?: string },
  left: GroupTimeline | null,
  right: GroupTimeline | null,
  includeThreadId = true,
): string | null {
  if (prev.threadId !== next.threadId || !prev.groupId || !next.groupId || !left || !right) return null;
  const leftIndex = left.indexById.get(prev.groupId);
  const rightIndex = right.indexById.get(next.groupId);
  if (leftIndex === undefined || rightIndex === undefined) return null;
  const page = pagingCall(prev.groupId, 'after', includeThreadId ? prev.threadId : undefined);
  // A carried group may also be in the later hit's home generation.
  const carriedIndex = right.indexById.get(prev.groupId);
  if (left.record.id === right.record.id && carriedIndex !== undefined) {
    const count = rightIndex - carriedIndex - 1;
    return count > 0 ? `— ${count} observation groups hidden between these results; continue with ${page} —` : null;
  }
  if (left.record.generationCount >= right.record.generationCount) return null;
  // Count only disjoint known groups. Never add overlapping carried groups twice.
  const laterIds = new Set(right.groups.map(group => group.id));
  const known = new Set(
    [
      ...left.groups.slice(leftIndex + 1).filter(group => !laterIds.has(group.id)),
      ...right.groups.slice(0, rightIndex).filter(group => {
        const index = left.indexById.get(group.id);
        return index === undefined || index > leftIndex;
      }),
    ].map(group => group.id),
  );
  const count = known.size ? `${known.size}+ observation groups hidden` : 'Additional observation groups may be hidden';
  return `— ${count} between these results; continue with ${page} —`;
}

export async function pageObservationGroups({
  om,
  threadId,
  resourceId,
  groupId,
  direction: requestedDirection,
  limit,
  threadTitle,
  includeThreadId = true,
}: {
  om: OMTimelineEngine;
  threadId: string;
  resourceId: string;
  groupId: string;
  direction?: 'before' | 'after';
  limit: number;
  threadTitle?: string;
  includeThreadId?: boolean;
}): Promise<{ results: string; count: number; hasMore?: boolean }> {
  const home = await findGroupTimeline(om, threadId, resourceId, groupId);
  if (!home)
    return {
      results: `No original observation group ${JSON.stringify(groupId)} was found in the active or buffered observations of thread ${JSON.stringify(threadId)} across its retained history. Check the threadId and groupId from the search hit, or use mode="messages" with a message ID from its source range.`,
      count: 0,
    };
  const direction = requestedDirection ?? 'after';
  let current = home;
  let position = home.indexById.get(groupId)! - (requestedDirection === undefined ? 1 : 0);
  const seen = new Set(
    (direction === 'after' ? home.groups.slice(0, position + 1) : home.groups.slice(position)).map(group => group.id),
  );
  const entries: Array<{ group: ObservationGroup; record: OMGenerationRecord }> = [];
  // One lookahead group lets us distinguish a full page from the end of retained history.
  while (entries.length <= limit) {
    position += direction === 'after' ? 1 : -1;
    const group = current.groups[position];
    if (group) {
      if (!seen.has(group.id)) {
        seen.add(group.id);
        entries.push({ group, record: current.record });
      }
      continue;
    }
    const generation = current.record.generationCount;
    const [record] = await om.getHistory(
      threadId,
      resourceId,
      1,
      direction === 'after'
        ? { afterGeneration: generation, sortDirection: 'ASC' }
        : { beforeGeneration: generation, sortDirection: 'DESC' },
    );
    if (!record) break;
    // Do not loop or expose other threads if a custom adapter ignores query options.
    if (
      (direction === 'after' ? record.generationCount <= generation : record.generationCount >= generation) ||
      (record.threadId !== null && record.threadId !== threadId)
    ) {
      throw new Error('Storage adapter did not honor observation history filters');
    }
    current = await buildTimeline(record, threadId);
    position = direction === 'after' ? -1 : current.groups.length;
  }
  const hasMore = entries.length > limit;
  const page = entries.slice(0, limit);
  if (direction === 'before') page.reverse();
  if (!page.length)
    return {
      results: `No ${direction === 'before' ? 'earlier' : 'later'} original observation groups in this thread's retained history.`,
      count: 0,
      hasMore: false,
    };
  const now = new Date();
  const text = page.map(({ group, record }) =>
    [
      `## Group \`${group.id}\``,
      `_range: \`${group.range}\`_`,
      addRelativeTimeToObservations(group.content, now, record.observedTimezone ?? undefined),
    ].join('\n'),
  );
  // Exclusive pages can always return toward their original anchor. Inclusive pages must check.
  const hasEarlier =
    direction === 'before'
      ? hasMore
      : requestedDirection !== undefined ||
        home.indexById.get(groupId)! > 0 ||
        (home.record.generationCount > 0 &&
          (await pageObservationGroups({ om, threadId, resourceId, groupId, direction: 'before', limit: 1 })).count >
            0);
  const hasLater = direction === 'before' || hasMore;
  const pagingThreadId = includeThreadId ? threadId : undefined;
  text.unshift(
    hasEarlier
      ? `— Browse earlier: ${pagingCall(page[0]!.group.id, 'before', pagingThreadId)} —`
      : '— Start of retained observation history for this thread. —',
  );
  text.push(
    hasLater
      ? `— Browse later: ${pagingCall(page.at(-1)!.group.id, 'after', pagingThreadId)} —`
      : '— End of retained observation history for this thread. —',
  );
  text.unshift(
    `### Observation page\nThread: ${JSON.stringify(threadTitle ?? threadId)}\nShowing ${page.length} groups ${requestedDirection === undefined ? 'starting at' : `strictly ${direction}`} \`${groupId}\` (oldest first).`,
  );
  return { results: text.join('\n\n'), count: page.length, hasMore };
}
