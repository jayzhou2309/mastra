---
'@mastra/memory': minor
---

Added observation-group paging to recall. Search results now display dates, chronological ordering, gap markers, and truncation guidance. A shared text allowance keeps every selected hit visible, with unused excerpt space assigned by relevance. Metadata and paging guidance remain outside the text allowance. Search uses compact references for excerpts or complete source ranges already in the current message list, freeing space for other hits. Later searches can show the text again after that context is removed. Agents can use `recall({ mode: "observations", groupId, direction: "after", limit: 5 })` to browse original groups across reflection generations without re-indexing existing records. Pages include indexed buffered observations before activation, without changing stored memory. Thread and anchor headers provide context, while `hasMore` and explicit boundary markers show when paging in a direction is complete.

OM now supplies system-level recall guidance even before the first observation, including in read-only runs. Reflected groups are labeled as lossy summaries. The guidance explains how to verify historical details, page around search hits, and retry an unsuccessful search with the user's message verbatim.
