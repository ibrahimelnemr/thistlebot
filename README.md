# Thistlebot

Thistlebot is a lightweight AI assistant with a small gateway service and CLI chat client. It supports Ollama, OpenRouter, and generic OpenAI-compatible providers, and stores config/prompts/session logs under `~/.thistlebot`.

## Run locally
1. Create and activate a virtual environment:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
2. Install in editable mode:
	```bash
	pip install -e .
	```
3. Initialize local state:
	```bash
	thistlebot setup
	```
   During setup, Thistlebot prompts for provider (`ollama`, `openrouter`, or `openai_compatible`), checks endpoint reachability, lists available models when possible, and asks you to select the primary model.
	Setup now also enables built-in tools and MCP defaults automatically:
	- `tools.runtime.enabled=true`
	- `tools.native.enabled=true`
	- `mcp.enabled=true`
	- `mcp.servers.open-websearch.enabled=true`
	If `npx` is missing, setup prints a warning to install Node.js and rerun setup (or run the MCP enable command after installation).
   Optional reset to defaults:
	```bash
	thistlebot reset
	```
4. Start the gateway (Terminal A):
	```bash
	thistlebot gateway
	```
5. Chat with the gateway (Terminal B):
	```bash
	thistlebot chat
	```
	Display controls:
	```bash
	thistlebot chat --render-markdown
	thistlebot chat --no-render-markdown
	```
	Streaming note: a `[loading...]` indicator appears before each assistant reply starts.

6. Start an agent-to-agent meeting (Terminal B):
	```bash
	thistlebot meeting
	```
	Meeting output notes:
	- `agent_a` and `agent_b` are colorized differently in terminal output.
	- Thinking blocks such as `<think>...</think>` are shown as `[thinking...]` and stream live as indented dim text.
	- A `[loading...]` indicator appears before each agent reply starts.
	- Replies render as markdown by default (disable with `--no-render-markdown`).
	Optional controls:
	```bash
	thistlebot meeting --max-turns 20
	thistlebot meeting --model-a qwen2.5:0.5b --model-b qwen2.5:0.5b
	thistlebot meeting --starter "are dogs better than cats"
	thistlebot meeting --no-render-markdown
	thistlebot meeting --system-a "Be playful and curious; always end with one open question."
	thistlebot meeting --system-b "Be analytical and skeptical; challenge one point and ask one follow-up question."
	```

Optional checks:
- Health endpoint: `curl http://127.0.0.1:7788/health`
- Ollama check: `thistlebot ollama check`
- Active provider check: `thistlebot llm check`
- GitHub login: `thistlebot github login`
- GitHub status: `thistlebot github status`
- GitHub repos: `thistlebot github repos --limit 20`

## Open Web Search MCP setup (for blogger research)

The blogger workflow can use MCP web search tools when MCP is enabled and
`open-websearch` is configured.

1. Install Node.js (which includes `npm` and `npx`).
2. Verify `npx` is available:
	 ```bash
	 npx --version
	 ```
3. Enable MCP and `open-websearch` in thistlebot config:
	 ```bash
	 thistlebot mcp enable open-websearch
	 ```
   Disable it later with:
   ```bash
   thistlebot mcp disable open-websearch
   ```
4. Verify the MCP server/tool visibility:
	 ```bash
	 thistlebot mcp status
	 thistlebot mcp tools
	 ```

Equivalent config shape in `~/.thistlebot/config.json`:

```json
{
	"mcp": {
		"enabled": true,
		"servers": {
			"open-websearch": {
				"enabled": true,
				"transport": "stdio",
				"command": "npx",
				"args": ["-y", "open-websearch@latest"],
				"env": {
					"MODE": "stdio"
				},
				"timeout_seconds": 30
			}
		}
	}
}
```

If enabled correctly, blogger research steps can invoke web search MCP tools to
gather fresher sources.

## Blogger agent scheduling and memory

The blogger agent supports scheduled autonomous runs using a local daemon.

1. Configure `~/.thistlebot/agents/blogger/config.json` schedule fields:
	- `schedule.enabled`: `true`
	- `schedule.cron`: cron expression (example: `0 9,21 * * *`)
	- Optional convenience fields: `schedule.times_per_day`, `schedule.interval_minutes`, or `schedule.interval_seconds`
2. Start the daemon:
	```bash
	thistlebot agent blogger
	```
	This runs in the foreground and streams scheduler execution in your terminal.
	For background mode, use:
	```bash
	thistlebot agent blogger start
	```
3. Check status and recent memory-backed summaries:
	```bash
	thistlebot agent blogger status -n 5
	```
4. Stop the daemon:
	```bash
	thistlebot agent blogger stop
	```

Run memories are stored in:

- `~/.thistlebot/agents/blogger/memory/index.json` (index of step/run entries)
- `~/.thistlebot/agents/blogger/runs/<run_id>/` (workflow artifacts)

## Blogger Agent Setup (Recommended)

Use the guided setup command:

```bash
thistlebot agent blogger setup
```

This setup command:

- ensures WordPress OAuth credentials/token are available,
- prompts for schedule preset (1-6/day, hourly, every 30 minutes, or custom cron),
- selects a WordPress site,
- chooses topic template (`ai`, `politics`, `finance`) or custom topic,
- saves runtime config and prints follow-up run commands.

Common non-interactive setup example:

```bash
thistlebot agent blogger setup \
	--site "your-site.wordpress.com" \
	--template ai \
	--post-status draft \
	--yes
```

Quick run after setup:

```bash
thistlebot agent blogger workflow post
```

Adjust runtime config values:

```bash
thistlebot agent blogger config show
thistlebot agent blogger config set topic="Latest AI tooling news" schedule.cron="0 0,6,12,18 * * *"
```

## Fast agent create

Create a new blogger-style agent instance with defaults:

```bash
thistlebot agent create --template politics --name myblog
```

Template sources are full agent skeletons under:

- `thistlebot/packs/blogging/agents/ai-blogger/`
- `thistlebot/packs/blogging/agents/politics-blogger/`
- `thistlebot/packs/blogging/agents/finance-blogger/`

Each template contains full framework artifacts (`AGENT.md`, `skills/*/SKILL.md`, workflows,
hooks, actions), and `agent create` copies those files into a new
`thistlebot/agents/<name>/` directory.

Zero-arg fast path:

```bash
thistlebot agent create
```

Defaults for `agent create`:

- template defaults to `ai`,
- autogenerated name defaults to `bloggerN` (first free index),
- schedule defaults to `00:00, 06:00, 12:00, 18:00` UTC,
- publish mode defaults to draft-safe behavior.

## Publish Reliability Notes

Why publish can fail:

- The LLM may generate a final answer without calling `wordpress.create_post`.
- External web/MCP timeouts can consume iterations before publish-tool usage.

Reliability measures in place:

- Skill-level hardening in `skills/publisher/SKILL.md` requiring explicit tool invocation.
- Guard-level enforcement requiring successful create-post tool result.
- Single-success guard blocks duplicate successful publish calls in one step.
- Draft enforcement keeps post status as `draft` unless explicitly configured for publish mode.
- Deterministic fallback in workflow execution: if publish step reaches final answer
	without a successful WordPress create-post call, Thistlebot attempts direct
	publish using resolved step inputs and the canonical `wordpress.create_post` tool.

Operational tips:

- Keep default `post_status` as `draft` until behavior is stable.
- Check `~/.thistlebot/agents/blogger/runs/<run_id>/events.jsonl` for tool-call
	traces when debugging publish failures.

## WordPress setup

Thistlebot uses WordPress.com API integration via:

```bash
thistlebot wordpress login
thistlebot wordpress sites
thistlebot wordpress test --yes
```

### Create a WordPress app (client id + client secret)

1. Open `https://developer.wordpress.com/apps/` and create a new app.
2. Set the redirect URI to exactly:
	```text
	http://127.0.0.1:8766/callback
	```
3. Save the app and copy the `client_id` and `client_secret`.

### Login from Thistlebot

```bash
thistlebot wordpress login \
  --client-id "<YOUR_CLIENT_ID>" \
  --client-secret "<YOUR_CLIENT_SECRET>" \
  --scope "posts sites media"
```

Then verify:

```bash
thistlebot wordpress status
thistlebot wordpress sites
```

Create a draft test post:

```bash
thistlebot wordpress test --yes
```

## Configure providers

The setup command writes provider configuration into `~/.thistlebot/config.json`:

```bash
thistlebot setup
```

### OpenRouter (recommended via env var)

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Example config:

```json
{
	"llm": {
		"provider": "openrouter",
		"model": "anthropic/claude-3.5-sonnet"
	},
	"providers": {
		"openrouter": {
			"base_url": "https://openrouter.ai/api/v1",
			"api_key_env": "OPENROUTER_API_KEY"
		}
	}
}
```

OpenRouter streaming behavior toggle:

```json
{
	"tools": {
		"runtime": {
			"openrouter_stream_with_tools": false
		}
	}
}
```

- `false` (default): bypasses tool-loop for streamed chat/meeting so token-level reasoning can stream live.
- `true`: enables tool-loop on OpenRouter streamed requests (you may lose token-by-token reasoning visibility because tool-loop currently resolves non-stream internally).

### Generic OpenAI-compatible endpoint

```bash
export OPENAI_API_KEY="your-key"
```

Example config:

```json
{
	"llm": {
		"provider": "openai_compatible",
		"model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
	},
	"providers": {
		"openai_compatible": {
			"base_url": "http://localhost:8000/v1",
			"api_key_env": "OPENAI_API_KEY"
		}
	}
}
```
