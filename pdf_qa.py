"""CLI: ask a question about a PDF and get a highlighted copy.

Usage:
    python pdf_qa.py path/to/file.pdf "What is the main argument of section 3?"
    python pdf_qa.py --model sonnet path/to/file.pdf "..."
"""
import argparse
import asyncio
import sys
from pathlib import Path

from agent_core import (
    DEFAULT_MODE,
    DEFAULT_MODEL,
    MODE_CHOICES,
    MODEL_CHOICES,
    run_agent,
)


async def main(pdf_path: str, question: str, model: str, mode: str) -> None:
    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")

    print(f"→ Asking Claude ({model}, {mode}) about {pdf.name}\n")

    async for kind, payload in run_agent(pdf, question, model=model, mode=mode):
        if kind == "text":
            print(payload, end="", flush=True)
        elif kind == "tool":
            print(f"\n  ⚙  {payload}", flush=True)
        elif kind == "done":
            cost = payload.get("cost_usd")
            cost_str = f" — cost ${cost:.4f}" if cost else ""
            print(f"\n\n✓ done{cost_str}")
            if payload.get("highlighted_pdf"):
                print(f"  Highlighted PDF: {payload['highlighted_pdf']}")
            cites = payload.get("citations") or []
            if cites:
                print(f"  Citations ({len(cites)}):")
                for c in cites:
                    mark = "  " if c.get("found", True) else " ✗"
                    print(f"   {mark} p.{c['page']}: {c['quote']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("question", nargs="+")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help=f"Claude model alias (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=MODE_CHOICES,
        help=f"Answer mode (default: {DEFAULT_MODE})",
    )
    args = ap.parse_args()
    asyncio.run(main(args.pdf_path, " ".join(args.question), args.model, args.mode))
