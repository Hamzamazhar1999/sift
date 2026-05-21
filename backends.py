"""Backend / model registry for sift.

This file is the single place to wire new LLMs into the app. Touching it
should not require any edits to `agent_core.py` — that module imports
`BACKENDS` and the resolver functions from here.

Two concepts:

1. A **backend** is the CLI binary the Agent SDK spawns (e.g. `claude`,
   `openclaude`). It defines what model aliases sift exposes for that CLI
   and which binary on disk to run.

2. A **profile** is an optional per-model override (only used by
   openclaude). It translates a sift-side alias like `gpt-4o-mini` into:
     - the real model id the CLI passes upstream
     - any extra CLI args (e.g. `--provider openai`)
     - env vars for the subprocess (e.g. `OPENAI_BASE_URL`)
     - which provider key file under `~/` to load

PROVIDER ROUTING NOTE — why everything is `provider: openai`:
openclaude's `--provider` accepts only `anthropic, openai, gemini,
github, bedrock, vertex, ollama`. There is no `openrouter` choice.
OpenRouter is reached by pointing openclaude's openai client at
OpenRouter's OpenAI-compatible base URL. So Gemini, GPT, and DeepSeek
all share `provider: openai` here — they differ only in the model slug
and they all auth with the same OPENROUTER_API_KEY.

ADDING A NEW MODEL:
- Same provider (more OpenRouter models): add an entry to `openclaude`'s
  `models` list and to `profiles` with the right `cli_model` slug.
- New native provider: add a new profile with `extra_args={"provider":
  "<name>"}`, drop the `OPENAI_BASE_URL` env override, and point
  `key_source` to whichever ~/<name>.key file the key lives in.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


BACKENDS: dict = {
    "claude": {
        "display": "Claude Code (Anthropic)",
        "binary": None,  # None → let the Agent SDK do its own discovery
        "models": ["haiku", "sonnet", "opus", "inherit"],
        "default_model": "haiku",
        "profiles": {},  # the Anthropic CLI doesn't need provider routing
    },
    "openclaude": {
        "display": "OpenClaude (OpenAI / Gemini / DeepSeek via OpenRouter)",
        "binary": "openclaude",
        "models": ["gpt-4o-mini", "gemini-2.5-flash", "deepseek-chat", "inherit"],
        "default_model": "gpt-4o-mini",
        "profiles": {
            "gpt-4o-mini": {
                "cli_model": "openai/gpt-4o-mini",
                "extra_args": {"provider": "openai"},
                "env": {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"},
                "key_source": "openrouter",
            },
            "gemini-2.5-flash": {
                "cli_model": "google/gemini-2.5-flash",
                "extra_args": {"provider": "openai"},
                "env": {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"},
                "key_source": "openrouter",
            },
            "deepseek-chat": {
                "cli_model": "deepseek/deepseek-chat-v3.1",
                "extra_args": {"provider": "openai"},
                "env": {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"},
                "key_source": "openrouter",
            },
        },
    },
}

BACKEND_CHOICES = tuple(BACKENDS.keys())
DEFAULT_BACKEND = "claude"

# Union of every backend's accepted model aliases — used purely for
# /ask body validation. The chosen CLI does the real check at spawn time.
MODEL_CHOICES = tuple(
    dict.fromkeys(m for b in BACKENDS.values() for m in b["models"])
)
DEFAULT_MODEL = BACKENDS[DEFAULT_BACKEND]["default_model"]


def resolve_cli_path(backend: str) -> str | None:
    """Look up the CLI binary for a backend. Returns None when the entry
    has no explicit binary (the SDK handles its own discovery in that
    case)."""
    binary = BACKENDS[backend].get("binary")
    if binary is None:
        return None
    path = shutil.which(binary)
    if not path:
        raise FileNotFoundError(
            f"backend `{backend}` expects the `{binary}` CLI on PATH "
            f"but couldn't find it. Install it and ensure `{binary}` "
            f"runs in your shell."
        )
    return path


def _load_provider_key(source: str) -> str:
    """Load a provider API key by name. Tries $<SOURCE>_API_KEY first, then
    falls back to ~/<source>.key. Returns the bare key (stripped). The
    caller is responsible for putting it in the subprocess env — we never
    log or print it."""
    env_name = f"{source.upper()}_API_KEY"
    val = os.environ.get(env_name)
    if val:
        return val.strip()
    key_file = Path.home() / f"{source}.key"
    if key_file.exists():
        return key_file.read_text().strip()
    raise FileNotFoundError(
        f"Need an API key for `{source}`. Set ${env_name} or create "
        f"~/{source}.key."
    )


def resolve_profile(backend: str, model: str):
    """Return (cli_model, extra_args, env) for the given backend+model.

    For backends without a profiles dict, or for models not present in the
    profiles dict, returns ((model or None), {}, {}) — meaning: pass the
    model alias straight through and add no extra args/env. This keeps the
    Anthropic `claude` backend on its existing path."""
    profile = BACKENDS[backend].get("profiles", {}).get(model)
    if not profile:
        # `inherit` means: don't pass --model at all (let the CLI choose).
        cli_model = None if model == "inherit" else model
        return cli_model, {}, {}
    env = dict(profile.get("env", {}))
    key_source = profile.get("key_source")
    if key_source:
        # openclaude's openai-provider path reads OPENAI_API_KEY. We feed
        # whichever provider's real key into that slot.
        env["OPENAI_API_KEY"] = _load_provider_key(key_source)
    return profile["cli_model"], dict(profile.get("extra_args", {})), env
