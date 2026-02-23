# Thistlebot

Thistlebot is a lightweight local AI assistant that wraps Ollama with a small gateway service and a CLI chat client. It stores config, prompts, and session logs under `~/.thistlebot` for simple local workflows.

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
   During setup, Thistlebot checks whether Ollama is reachable, lists available models, and prompts you to select the primary model.
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

6. Start an agent-to-agent meeting (Terminal B):
	```bash
	thistlebot meeting
	```
	Optional controls:
	```bash
	thistlebot meeting --max-turns 20
	thistlebot meeting --model-a qwen2.5:0.5b --model-b qwen2.5:0.5b
	thistlebot meeting --starter "Propose and refine a 3-step coding plan"
	```

Optional checks:
- Health endpoint: `curl http://127.0.0.1:7788/health`
- Ollama check: `thistlebot ollama check`
- GitHub login: `thistlebot github login`
- GitHub status: `thistlebot github status`
- GitHub repos: `thistlebot github repos --limit 20`
