import os
import pytest

from utils import llm


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"), reason="No LLM key configured")
def test_call_llm_smoke():
    # simple smoke test for call_llm: should return a non-empty string
    out = llm.call_llm("Say hello.", max_tokens=20)
    assert isinstance(out, str)
    assert len(out) > 0
