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
