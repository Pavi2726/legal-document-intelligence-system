"""
llm.py — FIXED VERSION
------------------------
Lightweight LLM wrapper for Legal Document Intelligence System.

FIXES APPLIED:
  FIX-01 : Groq now works with ONLY GROQ_API_KEY (no GROQ_API_URL needed)
           Uses official groq SDK directly and correctly
  FIX-02 : llm_analyze_risk response properly parsed —
           no more truncated "If any provision is deemed inval..."
  FIX-03 : llm_structured_extraction properly returns list of dicts —
           no more returning 0
  FIX-04 : Cleaner fallback chain: Groq → OpenAI → HuggingFace
  FIX-05 : max_tokens increased so responses don't get cut off
"""

import json
import logging
import os
from typing import List
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger(__name__)

_hf_generator = None

print("Loaded key:", os.getenv("GROQ_API_KEY"))
def _get_hf_generator():
    """Load HuggingFace fallback model — only used if no API key available."""
    global _hf_generator
    if _hf_generator is None:
        try:
            from transformers import pipeline
            model_name = os.getenv("HF_LLM_MODEL", "google/flan-t5-small")
            logger.info(f"Loading HF model: {model_name}")
            _hf_generator = pipeline("text2text-generation", model=model_name)
        except Exception as exc:
            logger.error(f"HF model load failed: {exc}")
            _hf_generator = None
    return _hf_generator


def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call an LLM with the given prompt.

    Priority order:
    1. Groq API (free, fast) — needs GROQ_API_KEY in .env
    2. OpenAI API           — needs OPENAI_API_KEY in .env
    3. HuggingFace local    — fallback, no key needed but slow

    FIX-01: Groq now only requires GROQ_API_KEY — no GROQ_API_URL needed.
    FIX-05: Default max_tokens increased to 1024 to prevent truncation.
    """

    # ── FIX-01: Groq — only needs API key ────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from groq import Groq

            client = Groq(api_key=groq_key)
            model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

            logger.info(f"Calling Groq model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"Groq response received: {len(result)} characters")
            return result

        except Exception as exc:
            logger.error(f"Groq call failed: {exc}")
            # Fall through to OpenAI

    # ── OpenAI fallback ───────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

            logger.info(f"Calling OpenAI model: {model}")

            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp["choices"][0]["message"]["content"].strip()

        except Exception as exc:
            logger.error(f"OpenAI call failed: {exc}")

    # ── HuggingFace local fallback ────────────────────────────────────
    gen = _get_hf_generator()
    if gen is None:
        raise RuntimeError(
            "No LLM backend available. "
            "Please set GROQ_API_KEY in your .env file. "
            "Get a free key at https://console.groq.com"
        )

    try:
        max_new = min(max_tokens, 256)
        out = gen(prompt, max_new_tokens=max_new, do_sample=False)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or first.get("text", "")).strip()
            return str(first).strip()
        return str(out)
    except Exception as exc:
        logger.error(f"HF generation failed: {exc}")
        raise


def _shorten_text(s: str, max_chars: int = 3500) -> str:
    """Trim text to fit within model context window."""
    return s if len(s) <= max_chars else s[:max_chars - 20] + "\n..."


def _extract_json_from_response(raw: str):
    """
    FIX-02 & FIX-03: Robustly extract JSON from LLM response.
    LLMs often wrap JSON in markdown code blocks like ```json ... ```
    This function strips those and parses cleanly.
    """
    if not raw:
        return None

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        cleaned = cleaned.split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1]
        cleaned = cleaned.split("```", 1)[0]

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        # Try to find JSON array or object in the text
        try:
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])
        except Exception:
            pass
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────
# Public helper functions
# ─────────────────────────────────────────────────────────────────────

def llm_summarize_text(text: str, top_k: int = 5) -> str:
    """
    Generate a concise bullet-point summary of the document text.
    """
    prompt = (
        "You are a legal assistant. Read the following legal document clauses "
        f"and produce a clear, concise summary with maximum {top_k} bullet points. "
        "Each bullet point should capture one key obligation, right, or restriction. "
        "Be specific — include amounts, dates, and party names where present.\n\n"
        f"DOCUMENT TEXT:\n{_shorten_text(text)}\n\n"
        "SUMMARY (bullet points):"
    )
    return call_llm(prompt, max_tokens=512)


def llm_analyze_risk(clauses: List[dict]) -> str:
    """
    FIX-02: Analyze legal risks across clauses.
    Returns properly formatted risk assessment — not truncated.
    """
    # Build clause payload
    payload = "\n\n".join(
        f"Clause {c.get('chunk_index', i)}: {c.get('content', '')}"
        for i, c in enumerate(clauses)
    )

    prompt = (
        "You are a legal risk analyst. Analyze the following legal clauses for risks.\n\n"
        "For each clause that contains a risk, provide:\n"
        "- Clause number\n"
        "- Risk description (what could go wrong)\n"
        "- Severity: LOW / MEDIUM / HIGH\n"
        "- Recommendation\n\n"
        "Return your response as a JSON array. Each object must have:\n"
        "chunk_index (int), risk (string), severity (string), recommendation (string)\n\n"
        f"CLAUSES:\n{_shorten_text(payload)}\n\n"
        "JSON RESPONSE:"
    )

    raw = call_llm(prompt, max_tokens=2048)

    # FIX-02: Try to parse as JSON first
    parsed = _extract_json_from_response(raw)
    if parsed is not None:
        return json.dumps(parsed, indent=2)

    # If not JSON, return the raw text (still useful, not truncated)
    return raw


def llm_structured_extraction(clauses: List[dict]) -> List[dict]:
    """
    FIX-03: Extract structured fields from clauses.
    Now properly returns a list of dicts — never returns 0.
    """
    payload = "\n\n".join(
        f"Clause {c.get('chunk_index', i)}: {c.get('content', '')}"
        for i, c in enumerate(clauses)
    )

    prompt = (
        "You are a legal data extraction assistant. "
        "Extract structured information from the following legal clauses.\n\n"
        "For EACH clause, return a JSON object with these exact keys:\n"
        "- chunk_index: the clause number (integer)\n"
        "- dates: list of all dates mentioned (strings)\n"
        "- amounts: list of all monetary amounts mentioned (strings)\n"
        "- parties: list of all party names or organization names mentioned (strings)\n\n"
        "Return a single JSON array containing one object per clause.\n"
        "If a field has no values, use an empty list [].\n\n"
        f"CLAUSES:\n{_shorten_text(payload)}\n\n"
        "JSON ARRAY:"
    )

    raw = call_llm(prompt, max_tokens=2048)

    # FIX-03: Robustly parse the response
    parsed = _extract_json_from_response(raw)

    if parsed is not None:
        # Make sure it's a list
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]

    # FIX-03: If LLM didn't return JSON, build basic structure from raw text
    logger.warning("LLM structured extraction did not return valid JSON — using fallback")
    return [
        {
            "chunk_index": c.get("chunk_index", i),
            "dates": [],
            "amounts": [],
            "parties": [],
            "note": "LLM response could not be parsed as JSON",
        }
        for i, c in enumerate(clauses)
    ]