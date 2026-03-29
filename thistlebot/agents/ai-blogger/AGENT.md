---
name: ai-blogger
version: 0.3.0
description: 'Template: autonomous AI-news blogger agent with iterative quality gates
  and WordPress publishing.'
tools:
- wordpress.*
- open-websearch.*
disallowedTools: []
model: null
x-thistlebot:
  config:
    defaults:
      topic: Latest AI news
      topic_template: ai
      post_status: draft
      publish_mode: draft
      enforce_draft_mode: true
      site: null
    required:
    - topic
    - site
    - post_status
  schedule:
    enabled: false
    cron: 0 9,21 * * *
    timezone: UTC
  workflow:
    default: daily_publish
    aliases:
      post: daily_publish
    overrides:
      research_max_iterations: 12
      draft_max_iterations: 10
      edit_max_iterations: 8
      verify_max_iterations: 6
      publish_max_iterations: 8
      max_revisions: 2
      verify_pass_token: 'VERDICT: PASS'
  hooks:
    pre_run:
    - type: idea_backlog_refresh
      enabled: true
      config:
        refresh_count: 6
        query_count: 8
        max_iterations: 14
        prefer_web: true
        min_refresh_interval_minutes: 180
    pre_topic_resolve:
    - type: idea_backlog_select
      enabled: true
      config: {}
    post_run:
    - type: idea_backlog_outcome
      enabled: true
      config:
        failure_selected_action: new
  actions:
    ideas-refresh:
      handler: hooks.idea_backlog:cli_refresh
      help: Refresh the idea backlog
    ideas-list:
      handler: hooks.idea_backlog:cli_list
      help: List backlog ideas
    ideas-select:
      handler: hooks.idea_backlog:cli_select
      help: Select one idea by id
---

Template: autonomous AI-news blogger agent with iterative quality gates and WordPress publishing.
