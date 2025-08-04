# MCP Integration Guide - nanpin_bot

## Issue Resolution: better-sqlite3 Dependency Error

### Problem
Claude Flow was failing with `Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'better-sqlite3'` when trying to use hooks and session management.

### Solution
The issue was that the `better-sqlite3` package was missing from the Claude Flow npx installation. Here's how it was resolved:

```bash
# 1. Find the Claude Flow npx installation directory
find ~/.npm/_npx -name "claude-flow" -type d

# 2. Install the missing dependency directly in the package
cd ~/.npm/_npx/7cfa166e65244432/node_modules/claude-flow
npm install better-sqlite3
```

### Verification
After installing the dependency, Claude Flow works correctly:

```bash
npx claude-flow@alpha hooks session-end --generate-summary true --persist-state true --export-metrics true
```

Output:
```
🔚 Executing session-end hook...
📊 Summary generation: ENABLED
💾 State persistence: ENABLED
📈 Metrics export: ENABLED
✅ ✅ Session-end hook completed
```

## Available MCP Resources

### ruv-swarm Server
- **Getting Started Guide**: `swarm://docs/getting-started`
- **Stability Features**: `swarm://docs/stability`

### Current System Status
- **Claude Flow Version**: v2.0.0-alpha.78
- **Node.js Version**: v22.17.1
- **npm Version**: 10.9.2

### WASM Module Status
```json
{
  "core": { "loaded": true, "size": 524288 },
  "neural": { "loaded": true, "size": 1048576 },
  "forecasting": { "loaded": true, "size": 1572864 },
  "swarm": { "loaded": false, "size": 786432 },
  "persistence": { "loaded": false, "size": 262144 }
}
```

### Available Features
- ✅ WebAssembly support
- ✅ SIMD support
- ✅ Neural networks (18 activation functions, 5 training algorithms)
- ✅ Forecasting (27 models available)
- ✅ Cognitive diversity (5 patterns available)
- ✅ Infinite runtime (no timeout mechanisms)

## Next Steps for MCP Integration

1. **Initialize Swarm**: Use `mcp__ruv-swarm__swarm_init` to set up coordination
2. **Spawn Agents**: Use `mcp__ruv-swarm__agent_spawn` for specialized agents
3. **Orchestrate Tasks**: Use `mcp__ruv-swarm__task_orchestrate` for complex workflows

## Best Practices

1. Always batch operations in single messages
2. Use memory for cross-agent coordination
3. Monitor progress with status tools
4. Take advantage of the infinite runtime feature (no timeouts)
5. Use the stability features for production deployments

## Session Management

The system now properly saves session state to `.swarm/memory.db` and tracks:
- Tasks completed
- Files edited
- Commands executed
- Success rates
- Duration metrics

This ensures continuity across sessions and enables learning from previous work.