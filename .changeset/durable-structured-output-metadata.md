---
'@mastra/core': patch
---

Fixed durable and evented agents not saving the structured output object on the assistant message in memory, so `content.metadata.structuredOutput` is now stored the same way as with a regular `Agent`. Fixes [#26432](https://github.com/mastra-ai/mastra/issues/26432).
