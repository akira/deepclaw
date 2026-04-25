#!/usr/bin/env python3
"""Run one DeepClaw eval case against a specific repo checkout and print JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--user-text", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace-env", required=True)
    return parser.parse_args()


async def run_once(agent, thread_id: str, input_messages: list[dict]):
    tool_calls_seen = False
    tool_names = []
    accumulated = ""
    async for chunk in agent.astream(
        {"messages": input_messages},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            continue
        message_obj, _metadata = chunk
        if isinstance(message_obj, ToolMessage):
            continue
        if not hasattr(message_obj, "content_blocks"):
            continue
        for block in message_obj.content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("tool_call", "tool_call_chunk"):
                tool_calls_seen = True
                name = block.get("name")
                if name:
                    tool_names.append(name)
            elif block_type == "text":
                accumulated += block.get("text", "")
    return {
        "tool_calls_seen": tool_calls_seen,
        "tool_names": tool_names,
        "text": accumulated,
    }


async def amain() -> None:
    args = parse_args()
    home_dir = tempfile.mkdtemp(prefix="deepclaw-eval-home-")
    os.environ["HOME"] = home_dir
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    load_dotenv(args.workspace_env, override=True)

    repo_root = Path(args.repo).resolve()
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    from deepclaw import gateway as gateway_mod
    from deepclaw.agent import create_agent, create_checkpointer
    from deepclaw.config import DeepClawConfig

    nudge_message = getattr(
        gateway_mod,
        "_NUDGE_MESSAGE",
        "You described an action but did not call any tools. Please call the appropriate tool now to carry out what you described.",
    )
    looks_like_narration = getattr(gateway_mod, "_looks_like_narration", lambda _text: False)
    looks_like_false_completion = getattr(
        gateway_mod, "_looks_like_false_completion", lambda _user, _assistant: False
    )
    looks_like_memory_request = getattr(
        gateway_mod, "_looks_like_memory_request", lambda _user, _assistant: False
    )

    config = DeepClawConfig(model=args.model, workspace_root=str(repo_root))
    async with create_checkpointer() as checkpointer:
        agent = create_agent(config, checkpointer)
        thread_id = f"eval-{uuid.uuid4()}"
        first = await run_once(agent, thread_id, [{"role": "user", "content": args.user_text}])
        attempts = 1
        retried = False
        final = first
        if (not first["tool_calls_seen"]) and (
            looks_like_narration(first["text"])
            or looks_like_false_completion(args.user_text, first["text"])
            or looks_like_memory_request(args.user_text, first["text"])
        ):
            retried = True
            attempts += 1
            second = await run_once(agent, thread_id, [{"role": "user", "content": nudge_message}])
            final = {
                "tool_calls_seen": second["tool_calls_seen"],
                "tool_names": first["tool_names"] + second["tool_names"],
                "text": second["text"],
            }
        sys.stdout.write(
            json.dumps(
                {
                    "final_text": final["text"],
                    "tool_calls_seen": final["tool_calls_seen"] or first["tool_calls_seen"],
                    "tool_names": final["tool_names"],
                    "attempts": attempts,
                    "retried": retried,
                    "first_pass_tool_calls_seen": first["tool_calls_seen"],
                    "first_pass_text": first["text"],
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    asyncio.run(amain())
