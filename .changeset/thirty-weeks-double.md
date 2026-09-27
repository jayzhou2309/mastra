---
'@mastra/core': minor
'@mastra/libsql': minor
'@mastra/pg': minor
'@mastra/mysql': minor
'@mastra/mongodb': minor
'@mastra/oracledb': minor
'@mastra/convex': minor
---

Added group filtering and generation ordering to observational memory history. For example, `getObservationalMemoryHistory(threadId, resourceId, 1, { groupId, sortDirection: "ASC" })` finds the earliest retained record containing a group in active observations or persisted buffered chunks. Adapters advertise support through `supportsObservationalMemoryHistorySearch`.
