"""Send prompts to the OpenAI API and retrieve responses."""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

RAW_LOG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "raw_responses"

@dataclass
class LLMConfig:
    """Configuration for the LLM API call."""
    model: str = "gpt-4.1"
    temperature: float = 0.0 # deterministic output for evaluation
    max_tokens: int = 4096

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found"
            )
        _client = OpenAI(api_key=api_key)
    return _client

def _save_raw_log(
    prompt: str,
    response_text: str,
    config: LLMConfig,
    step: str = "",
    project_id: str = "",
    is_pilot: bool = False,
) -> Path:
    """Save prompt + raw model output so runs are easier to trace later"""
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{step}_{project_id}_{timestamp}.json" if step else f"raw_{timestamp}.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": config.model,
        "max_tokens": config.max_tokens,
        "step": step,
        "project_id": project_id,
        "is_pilot": is_pilot,
        "prompt": prompt,
        "raw_response": response_text,
    }

    path = RAW_LOG_DIR / filename
    path.write_text(json.dumps(log_entry, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def send_prompt(
    prompt: str,
    config: LLMConfig | None = None,
    step: str = "",
    project_id: str = "",
    is_pilot: bool = False,
) -> str:
    """Send a prompt to the OpenAI API and return the text response"""
    cfg = config or LLMConfig()
    client = _get_client()

    # Status
    label = f"{step or 'llm'}" + (f" project={project_id}" if project_id else "")
    sys.stdout.write(f"  [ ] Waiting for {cfg.model} ({label})...")
    sys.stdout.flush()

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        sys.stdout.write(f"\r  [❌] Failed    {cfg.model} ({label})\n")
        sys.stdout.flush()
        raise

    elapsed = time.perf_counter() - start
    sys.stdout.write(f"\r  [✅] Done in {elapsed:5.1f}s  {cfg.model} ({label})\n")
    sys.stdout.flush()


    response_text = response.choices[0].message.content

    # Save raw prompt + response for reproducibility
    log_path = _save_raw_log(prompt, response_text, cfg, step, project_id, is_pilot)
    logger.info("Saved raw LLM log to %s", log_path)

    return response_text