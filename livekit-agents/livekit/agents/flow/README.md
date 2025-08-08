Flow Module Notes

- Transition equations support operators: ==, !=, >, <, >=, <=, contains, startswith, endswith, in, ==~ (case-insensitive equals), !=~ (case-insensitive not equals), matches, matches_i.
- Operand resolution:
  - JSONPath-like operands ($, @, ., [, ], *) are evaluated against the evaluation context.
  - Simple identifiers resolve from the local context, then from flow variables.
  - If nothing resolves, the operand is treated as a literal string.
- Booleans and numbers on the right-hand side are coerced from strings: "true", "false", "1", "0", numeric forms.

Global Triggers
- Conversation nodes with global_node_setting are synthesized as prompt-based edges during user input evaluation.
- These transitions are only evaluated when user_text is present.

Function Tools (ToolExecutor)
- Local tools (type: local) do not create HTTP sessions.
- Custom tools (type: custom) use aiohttp with retry and exponential backoff (with jitter).
- Circuit breaker can be enabled via cb_max_failures and cb_reset_ms.
- Per-tool concurrency limit via max_concurrency (integer). When set, concurrent executions for the same tool_id are limited accordingly.

Examples
- Minimal: examples/flows/minimal_test_flow.json
- Advanced: examples/flows/advanced_test_flow.json (adds branching, gather validation, multiple functions, and global handoff trigger)
