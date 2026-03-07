# Plan: Implementing local OpenClaw Orchestration (Plan Skill)

This document outlines the migration of the `everyday_my_arxiv` pipeline from GitHub Actions to a local OpenClaw Agent Skill.

## Goal
Replace the rigid GitHub Action with a granular, state-persistent local orchestration that allows for manual intervention, robust retries, and better environment integration.

## 1. Orchestration Structure
The "Plan Skill" will wrap the `scripts/run_daily_report.py` logic into a managed OpenClaw task.

### Stages
- **Stage 1: Fetch** (Arxiv API)
- **Stage 2: Score** (Relevance filtering)
- **Stage 3: Analyze** (Deep dive with LLM)
- **Stage 4: Report** (Markdown generation & Notification)

## 2. Technical Implementation
- **Skill Name:** `everyday-my-arxiv`
- **Location:** `~/.openclaw/workspace/skills/everyday-my-arxiv/`
- **Logic:**
    - Use `python3 scripts/run_daily_report.py --stage <stage>` for execution.
    - Monitor `state.json` (using the custom encoders/decoders) to track progress.
    - If a stage fails, the skill will persist the state and alert Owen via Feishu/Desktop Notification.

## 3. Automation & Schedule
- **Trigger:** Cron job running at 09:00 JST every weekday.
- **Heartbeat:** Check `state.json` during heartbeat turns to ensure no pipeline is "stuck".

## 4. Error Handling Protocol
- **Arxiv Timeout:** Retry with exponential backoff.
- **LLM Quota/Error:** Fallback between Gemini (Flash) and Ark (V3/V4).
- **Git Sync:** Automatic `git pull --rebase` before reporting to ensure the local repo is synchronized with any manual keyword updates.

## 5. Notification Strategy
- **Start:** `notify-send "Arxiv Insight" "Starting daily research scan..."`
- **Progress:** Updates via Feishu for significant findings.
- **Completion:** Final report link sent to Feishu and a desktop notification.

---
**Next Step:** Initialize the skill folder and link the entry points.
